"""推論：シナリオ別の周辺分布、観測によるベイズ更新、要因の逆算、介入と因果効果、感度分析、einsum による検算。"""
from __future__ import annotations

import warnings

import numpy as np
from fxtensor_salmon import FXTensor

from . import network
from .causal import do_config, do_vector, effect_spec, sensitivity, sensitivity_specs
from .config import config_hash
from .network import Net, self_test_graph_extend, wiring, wiring_size

MAX_JOINT = 2_000_000      # 同時分布の要素数がこれを超えたら祖先集合ごとの計算に切り替え


def _r(x, nd):
    """丸めて -0.0 を 0.0 に正規化する。"""
    x = round(float(x), nd)
    return 0.0 if x == 0 else x


def as_dict(labels, arr, nd=4):
    return {lab: _r(x, nd) for lab, x in zip(labels, arr)}


def movement(prior, post, labels):
    """事前→事後の動き（差・全変動距離・KL）。"""
    p = np.asarray(prior, float)
    q = np.array([post[lab] for lab in labels])
    nz = q > 0
    return {
        "差(pt)": {lab: _r((q[i] - p[i]) * 100, 1) for i, lab in enumerate(labels)},
        "全変動距離TV": _r(0.5 * np.abs(q - p).sum(), 4),
        "KL(事後||事前)[nat]": _r(np.sum(q[nz] * np.log(q[nz] / np.maximum(p[nz], 1e-300))), 4),
    }


def evidence_vectors(net, evidence):
    """証拠 {変数: ラベル | {"likelihood": [...]} | [...]} を {変数: 尤度ベクトル} にする。"""
    out = {}
    for v, e in evidence.items():
        labs = net.labels[v]
        if isinstance(e, str):
            out[v] = np.array([1.0 if lab == e else 0.0 for lab in labs])
        else:
            out[v] = np.asarray(e["likelihood"] if isinstance(e, dict) else e, float)
    return out


def normalize_observation(ob):
    """observations の 1 項目を (結果のキー, 証拠の辞書, 旧形式か) にする。"""
    if "evidence" in ob:
        ev = ob["evidence"]
    else:
        ev = {ob["var"]: ob["label"]}

    def show(v, e):
        if isinstance(e, str):
            return f"{v}={e}"
        lik = e["likelihood"] if isinstance(e, dict) else e
        return f"{v}~L{[float(x) for x in lik]}"

    key = ob.get("name") or (", ".join(show(v, e) for v, e in ev.items()) + " を観測")
    simple = len(ev) == 1 and isinstance(next(iter(ev.values())), str)
    return key, ev, simple


class Engine:
    def __init__(self, net, priors, max_joint=MAX_JOINT):
        self.net, self.priors = net, priors
        self.mode = "joint" if net.joint_size() <= max_joint else "ancestral"
        self.check_err = 0.0
        if self.mode == "joint":
            self.joint, self.names = net.build_joint(priors)
            self.labs = [net.labels[v] for v in self.names]
            # build_joint の軸は「根→子」の順、brute_force_joint の軸は net.order の順。
            # 後者を前者の順に並べ替えて比べる（根を子より後ろに宣言しても正しく比べられる）。
            perm = [net.order.index(v) for v in self.names]
            self.check_err = float(np.max(np.abs(self.joint.data - net.brute_force_joint(priors).transpose(perm))))

    def _wiring_fits(self, names):
        """配線核（密）が上限 network.MAX_WIRING に収まるか。"""
        return wiring_size(self.labs, [self.names.index(v) for v in names]) <= network.MAX_WIRING

    def _sum_joint(self, names):
        """ライブラリで組んだ同時分布を、names 以外の軸について直接和をとる（配線核を作らない）。"""
        data = self.joint.data
        idx = [self.names.index(v) for v in names]
        rest = [j for j in range(data.ndim) if j not in idx]
        # 和をとる軸を末尾に集めて連続配列にし、1 軸の和（numpy のペアワイズ加算）で丸め誤差を抑える
        flat = np.ascontiguousarray(np.transpose(data, idx + rest)).reshape([data.shape[i] for i in idx] + [-1])
        return flat.sum(axis=-1)

    def marginal(self, v):
        """周辺分布。joint: ライブラリの配線核で discard（配線核が上限を超えるときは同時分布の直接和）。
        ancestral: 祖先集合の einsum（変数消去）。"""
        if self.mode == "joint":
            if self._wiring_fits([v]):
                return self.joint.composition(wiring(self.labs, [self.names.index(v)])).data
            return self._sum_joint([v])
        return self.net.einsum_query(self.priors, [v])

    def pair(self, a, b):
        """I → a ⊗ b の同時状態（FXTensor）。配線核が上限を超えるときは同時分布の直接和（joint）か einsum（ancestral）。"""
        net = self.net
        if self.mode == "joint":
            if self._wiring_fits([a, b]):
                return self.joint.composition(wiring(self.labs, [self.names.index(a), self.names.index(b)]))
            data = self._sum_joint([a, b])
        else:
            data = net.einsum_query(self.priors, [a, b])
        return FXTensor([[], [net.labels[a], net.labels[b]]], data=data)

    def condition_raw(self, obs, label):
        """観測 obs=label の下での各変数の事後（丸め前の ndarray）。(obs, v) の2変数状態を conditionalization(2)。"""
        net = self.net
        ev = float(self.marginal(obs)[net.labels[obs].index(label)])
        if ev <= 0:
            raise ValueError(f"観測 {obs}={label} の確率が 0 のため条件付けできません")
        post = {}
        for v in net.order:
            if v == obs:
                continue
            cond = self.pair(obs, v).conditionalization(2)            # obs → v
            post[v] = np.asarray(cond.data[cond.get_label_index(0, label)], float)
        return ev, post

    def condition(self, obs, label):
        """condition_raw を小数4桁に丸めた辞書で返す。"""
        ev, raw = self.condition_raw(obs, label)
        return ev, {v: as_dict(self.net.labels[v], a) for v, a in raw.items()}

    def posterior_raw(self, evidence, targets=None):
        """複数の証拠（ハード・ソフト＝尤度）の同時の条件付け。変数消去（祖先集合の einsum）で計算する。

        evidence: {変数: ラベル | {"likelihood": [...]}}。戻り値: (証拠の確率（尤度の期待値）, {変数: 事後})。
        targets を省略すると、ハード証拠の変数以外のすべての変数。
        """
        net = self.net
        lik = evidence_vectors(net, evidence)
        ev = float(net.contract(self.priors, [], evidence=lik))
        if not ev > 0:
            raise ValueError(f"証拠 {evidence} の確率が 0 のため条件付けできません")
        hard = {v for v, e in evidence.items() if isinstance(e, str)}
        targets = targets or [v for v in net.order if v not in hard]
        return ev, {v: net.contract(self.priors, [v], evidence=lik) / ev for v in targets}

    def inversion_roots(self, obs, label):
        """inversion_roots_raw を小数4桁に丸めた辞書で返す。"""
        raw = self.inversion_roots_raw(obs, label)
        return {r: as_dict(self.net.labels[r], a) for r, a in raw.items()}

    def inversion_size(self, obs):
        """要因の逆算で作る密テンソル（根の全体 × obs）の要素数。"""
        net = self.net
        return int(np.prod([len(net.labels[r]) for r in net.roots])) * len(net.labels[obs])

    def inversion_roots_raw(self, obs, label):
        """核 (根の全体)→obs を einsum で作り FXTensor.bayesian_inversion で根の事後を出す（要因の逆算、丸め前）。"""
        net = self.net
        kdata = net.einsum_query(self.priors, [obs], free_roots=True)
        k = FXTensor([[net.labels[r] for r in net.roots], [net.labels[obs]]], data=kdata)
        assert k.is_markov()
        inv = k.bayesian_inversion(net.prior_state(self.priors))
        row = inv.data[inv.get_label_index(0, label)]
        out = {}
        for i, r in enumerate(net.roots):
            axes = tuple(j for j in range(len(net.roots)) if j != i)
            out[r] = np.asarray(row.sum(axis=axes) if axes else row, float)
        return out


def _evidence_json(ev):
    return {v: e if isinstance(e, str) else {"尤度": [float(x) for x in (e["likelihood"] if isinstance(e, dict) else e)]}
            for v, e in ev.items()}


def run(cfg, max_joint=MAX_JOINT):
    """設定を計算して結果の辞書を返す。検算で食い違いがあれば AssertionError。"""
    self_test_graph_extend()
    net = Net(cfg)
    base_pri = {v: cfg["priors"][v]["p"] for v in net.roots}
    res = {
        "基準日": cfg["as_of"], "ホライズン": cfg.get("horizon"), "注意": cfg.get("note"),
        "設定ハッシュ": config_hash(cfg),
        "事前分布(基本)": {v: as_dict(net.labels[v], base_pri[v]) for v in net.roots},
        "シナリオ": {}, "ベイズ更新(基本)": {}, "検算": {},
    }
    checks = {"計算モード": None, "同時分布の要素数": net.joint_size(),
              "einsum(全同時)との最大差": 0.0, "einsum(祖先集合・周辺)との最大差": 0.0,
              "einsum(条件付け)との最大差": 0.0, "bayesian_inversionとconditionalizationの最大差": 0.0}

    def upd(key, err):
        checks[key] = max(checks.get(key, 0.0), float(err))

    outputs = cfg.get("outputs") or [v for v in net.order if v not in net.roots]
    for name, ov in {"基本": {}, **cfg.get("scenarios", {})}.items():
        pri = {**base_pri, **ov}
        eng = Engine(net, pri, max_joint)
        checks["計算モード"] = eng.mode
        upd("einsum(全同時)との最大差", eng.check_err)
        marg = {}
        for v in outputs:
            m = eng.marginal(v)
            upd("einsum(祖先集合・周辺)との最大差", np.max(np.abs(m - net.einsum_query(pri, [v]))))
            marg[v] = as_dict(net.labels[v], m)
        res["シナリオ"][name] = {"上書き": ov, "周辺": marg}

    eng = Engine(net, base_pri, max_joint)
    skipped_inv = []
    for ob in cfg.get("observations", []):
        key, evidence, simple = normalize_observation(ob)
        # ソフト証拠を含むと ev は確率ではなく尤度の期待値 Σ P(x) L(x)（1 を超えうる）なので別のキーにする
        ev_key = "観測前の確率" if all(isinstance(e, str) for e in evidence.values()) else "尤度の期待値"
        focus = ob.get("focus") or net.roots
        if not float(net.contract(base_pri, [], evidence=evidence_vectors(net, evidence))) > 0:
            # 確率 0 の観測は全体を止めず、その観測の更新だけ省いて結果に記録する
            warnings.warn(f"観測「{key}」は確率 0 のため条件付けできません（この観測の更新は省略します）",
                          stacklevel=2)
            entry = {ev_key: 0.0, "不可能な観測": True, "事後": {},
                     "注目変数の事前": {w: as_dict(net.labels[w], eng.marginal(w)) for w in focus},
                     "注目変数の動き": {}, "動きの大きさ順(TV)": []}
            res["ベイズ更新(基本)"][key] = entry if simple else {"証拠": _evidence_json(evidence), **entry}
            continue
        if simple:
            (v, lab), = evidence.items()
            ev, raw = eng.condition_raw(v, lab)
            li = net.labels[v].index(lab)
            for w, a in raw.items():           # einsum 検算（祖先集合で (obs, w) を直接）。丸め前の値で比べる
                pr = net.einsum_query(base_pri, [v, w])
                ref = pr[li] / pr[li].sum()
                upd("einsum(条件付け)との最大差", np.max(np.abs(a - ref)))
            if eng.inversion_size(v) <= network.MAX_WIRING:
                inv = eng.inversion_roots_raw(v, lab)
                for r, a in inv.items():
                    if r != v:
                        upd("bayesian_inversionとconditionalizationの最大差", np.max(np.abs(a - raw[r])))
            else:      # 根の直積が大きすぎるときは逆算の検算を省く（密テンソルを作らない）
                skipped_inv.append(key)
        else:
            ev, raw = eng.posterior_raw(evidence)
            if eng.mode == "joint":            # 検算：全同時分布に尤度を掛けて直接周辺化
                lik = evidence_vectors(net, evidence)
                J = net.brute_force_joint(base_pri)
                for w, vec in lik.items():
                    shape = [1] * J.ndim
                    shape[net.order.index(w)] = len(vec)
                    J = J * vec.reshape(shape)
                Z = J.sum()
                upd("einsum(条件付け)との最大差", abs(Z - ev))
                for w, a in raw.items():
                    i = net.order.index(w)
                    ref = J.sum(axis=tuple(j for j in range(J.ndim) if j != i)) / Z
                    upd("einsum(条件付け)との最大差", np.max(np.abs(a - ref)))
                del J
        post = {w: as_dict(net.labels[w], a) for w, a in raw.items()}
        hard = {w for w, e in evidence.items() if isinstance(e, str)}
        prior_marg = {w: eng.marginal(w) for w in focus}
        mv = {w: movement(prior_marg[w], post[w], net.labels[w]) for w in focus if w not in hard}
        ranking = sorted(mv, key=lambda w: -mv[w]["全変動距離TV"])
        entry = {ev_key: round(ev, 4), "事後": post,
                 "注目変数の事前": {w: as_dict(net.labels[w], prior_marg[w]) for w in focus},
                 "注目変数の動き": mv, "動きの大きさ順(TV)": ranking}
        if not simple:
            entry = {"証拠": _evidence_json(evidence), **entry}
        res["ベイズ更新(基本)"][key] = entry
    if skipped_inv:
        checks["bayesian_inversionの検算を省略した観測"] = skipped_inv

    # --- 介入（グラフ手術）---
    def do_marginals(do, targets):
        """do の下での targets の周辺（派生設定の Engine＝ライブラリで計算し、元のネットの切断因子分解と照合）。"""
        dcfg = do_config(cfg, do)
        dnet = Net(dcfg)
        deng = Engine(dnet, {q: dcfg["priors"][q]["p"] for q in dnet.roots}, max_joint)
        upd("einsum(全同時)との最大差", deng.check_err)
        dvec = {v: do_vector(net.labels[v], s) for v, s in do.items()}
        out = {}
        for t in targets:
            m = deng.marginal(t)
            upd("介入:切断因子分解との最大差", np.max(np.abs(m - net.contract(base_pri, [t], do=dvec))))
            out[t] = m
        return out

    if cfg.get("interventions"):
        res["介入"] = {}
        for name, do in cfg["interventions"].items():
            m = do_marginals(do, outputs)
            res["介入"][name] = {"介入内容": {v: as_dict(net.labels[v], do_vector(net.labels[v], s)) for v, s in do.items()},
                               "周辺": {t: as_dict(net.labels[t], a) for t, a in m.items()}}
    if cfg.get("effects"):
        res["因果効果"] = {}
        for e in cfg["effects"]:
            e = effect_spec(cfg, e)
            X, Y, y = e["treatment"], e["outcome"], e["outcome_label"]
            yi = net.labels[Y].index(y)
            p_do, p_obs = {}, {}
            for x in net.labels[X]:
                p_do[x] = float(do_marginals({X: x}, [Y])[Y][yi])
                lik = evidence_vectors(net, {X: x})
                z = float(net.contract(base_pri, [], evidence=lik))
                p_obs[x] = float(net.contract(base_pri, [Y], evidence=lik)[yi] / z) if z > 0 else None
            t_, c_ = e["treated"], e["control"]
            ace = p_do[t_] - p_do[c_]
            od = None if p_obs[t_] is None or p_obs[c_] is None else p_obs[t_] - p_obs[c_]
            res["因果効果"][f"{X} → {Y}={y}"] = {
                "処置": X, "結果": Y, "結果のラベル": y, "処置群": t_, "対照群": c_,
                "P(結果|do(処置))": {x: _r(p, 4) for x, p in p_do.items()},
                "P(結果|処置)(観測)": {x: None if p is None else _r(p, 4) for x, p in p_obs.items()},
                "ACE": _r(ace, 4),
                "観測上の差": None if od is None else _r(od, 4),
                "交絡による差(観測上の差−ACE)": None if od is None else _r(od - ace, 4),
            }
    specs = sensitivity_specs(cfg)
    if specs:
        res["感度分析"] = {}
        for s in specs:
            base, n, rows = sensitivity(net, base_pri, s["target"], s["label"], s["delta"], s["top"])
            res["感度分析"][f"{s['target']}={s['label']}"] = {
                "目的": s["target"], "ラベル": s["label"], "基準のP": _r(base, 4), "変化量delta": s["delta"],
                "パラメータ数": n,
                "上位": [{k: (_r(v, 4) if k != "幅(pt)" else _r(v, 2)) if isinstance(v, float) else v
                         for k, v in r.items()} for r in rows],
            }

    for k in list(checks):
        if isinstance(checks[k], float):
            checks[k] = float(f"{checks[k]:.3g}")
    checks["備考"] = "検算はすべて丸め前の値で比較（出力の確率は小数4桁に丸めている）"
    res["検算"] = checks
    assert checks["einsum(全同時)との最大差"] < 1e-12 and checks["einsum(祖先集合・周辺)との最大差"] < 1e-12
    assert checks["einsum(条件付け)との最大差"] < 1e-9 and checks["bayesian_inversionとconditionalizationの最大差"] < 1e-9
    assert checks.get("介入:切断因子分解との最大差", 0.0) < 1e-12
    return res
