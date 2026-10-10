"""推論：シナリオ別の周辺分布、観測によるベイズ更新、要因の逆算、einsum による検算。"""
from __future__ import annotations

import numpy as np
from fxtensor_salmon import FXTensor

from .config import config_hash
from .network import Net, self_test_graph_extend, wiring

MAX_JOINT = 2_000_000      # 同時分布の要素数がこれを超えたら祖先集合ごとの計算に切り替え


def as_dict(labels, arr, nd=4):
    return {lab: round(float(x), nd) for lab, x in zip(labels, arr)}


def movement(prior, post, labels):
    """事前→事後の動き（差・全変動距離・KL）。"""
    p = np.asarray(prior, float)
    q = np.array([post[lab] for lab in labels])
    nz = q > 0
    return {
        "差(pt)": {lab: round(float((q[i] - p[i]) * 100), 1) for i, lab in enumerate(labels)},
        "全変動距離TV": round(float(0.5 * np.abs(q - p).sum()), 4),
        "KL(事後||事前)[nat]": round(float(np.sum(q[nz] * np.log(q[nz] / np.maximum(p[nz], 1e-300)))), 4),
    }


class Engine:
    def __init__(self, net, priors, max_joint=MAX_JOINT):
        self.net, self.priors = net, priors
        self.mode = "joint" if net.joint_size() <= max_joint else "ancestral"
        self.check_err = 0.0
        if self.mode == "joint":
            self.joint, self.names = net.build_joint(priors)
            self.labs = [net.labels[v] for v in self.names]
            perm = [net.order.index(v) for v in self.names]
            self.check_err = float(np.max(np.abs(
                np.transpose(self.joint.data, perm) - net.brute_force_joint(priors).transpose(perm)
            )))

    def marginal(self, v):
        """周辺分布（joint: ライブラリの配線核で discard / ancestral: 祖先集合の einsum）。"""
        if self.mode == "joint":
            return self.joint.composition(wiring(self.labs, [self.names.index(v)])).data
        return self.net.einsum_query(self.priors, [v])

    def pair(self, a, b):
        """I → a ⊗ b の同時状態（FXTensor）。"""
        net = self.net
        if self.mode == "joint":
            return self.joint.composition(wiring(self.labs, [self.names.index(a), self.names.index(b)]))
        return FXTensor([[], [net.labels[a], net.labels[b]]], data=net.einsum_query(self.priors, [a, b]))

    def condition(self, obs, label):
        """観測 obs=label の下での各変数の事後（(obs, v) の2変数状態を conditionalization(2)）。"""
        net = self.net
        ev = float(self.marginal(obs)[net.labels[obs].index(label)])
        if ev <= 0:
            raise ValueError(f"観測 {obs}={label} の確率が 0 のため条件付けできません")
        post = {}
        for v in net.order:
            if v == obs:
                continue
            cond = self.pair(obs, v).conditionalization(2)            # obs → v
            post[v] = as_dict(net.labels[v], cond.data[cond.get_label_index(0, label)])
        return ev, post

    def inversion_roots(self, obs, label):
        """核 (根の全体)→obs を einsum で作り FXTensor.bayesian_inversion で根の事後を出す（要因の逆算）。"""
        net = self.net
        kdata = net.einsum_query(self.priors, [obs], free_roots=True)
        k = FXTensor([[net.labels[r] for r in net.roots], [net.labels[obs]]], data=kdata)
        assert k.is_markov()
        inv = k.bayesian_inversion(net.prior_state(self.priors))
        row = inv.data[inv.get_label_index(0, label)]
        out = {}
        for i, r in enumerate(net.roots):
            axes = tuple(j for j in range(len(net.roots)) if j != i)
            out[r] = as_dict(net.labels[r], row.sum(axis=axes) if axes else row)
        return out


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
    outputs = cfg.get("outputs") or [v for v in net.order if v not in net.roots]
    for name, ov in {"基本": {}, **cfg.get("scenarios", {})}.items():
        pri = {**base_pri, **ov}
        eng = Engine(net, pri, max_joint)
        checks["計算モード"] = eng.mode
        checks["einsum(全同時)との最大差"] = max(checks["einsum(全同時)との最大差"], eng.check_err)
        marg = {}
        for v in outputs:
            m = eng.marginal(v)
            e = net.einsum_query(pri, [v])
            checks["einsum(祖先集合・周辺)との最大差"] = max(checks["einsum(祖先集合・周辺)との最大差"],
                                                       float(np.max(np.abs(m - e))))
            marg[v] = as_dict(net.labels[v], m)
        res["シナリオ"][name] = {"上書き": ov, "周辺": marg}

    eng = Engine(net, base_pri, max_joint)
    for ob in cfg.get("observations", []):
        v, lab = ob["var"], ob["label"]
        ev, post = eng.condition(v, lab)
        for w, d in post.items():          # einsum 検算（祖先集合で (obs, w) を直接）
            pr = net.einsum_query(base_pri, [v, w])
            ref = pr[net.labels[v].index(lab)] / pr[net.labels[v].index(lab)].sum()
            checks["einsum(条件付け)との最大差"] = max(checks["einsum(条件付け)との最大差"],
                                                   float(np.max(np.abs(np.array(list(d.values())) - ref))))
        inv = eng.inversion_roots(v, lab)
        for r, d in inv.items():
            if r != v:
                checks["bayesian_inversionとconditionalizationの最大差"] = max(
                    checks["bayesian_inversionとconditionalizationの最大差"],
                    max(abs(d[lab_] - post[r][lab_]) for lab_ in d))
        focus = ob.get("focus") or net.roots
        prior_marg = {w: eng.marginal(w) for w in focus}
        mv = {w: movement(prior_marg[w], post[w], net.labels[w]) for w in focus if w != v}
        ranking = sorted(mv, key=lambda w: -mv[w]["全変動距離TV"])
        res["ベイズ更新(基本)"][f"{v}={lab} を観測"] = {
            "観測前の確率": round(ev, 4), "事後": post,
            "注目変数の事前": {w: as_dict(net.labels[w], prior_marg[w]) for w in focus},
            "注目変数の動き": mv, "動きの大きさ順(TV)": ranking,
        }
    for k in list(checks):
        if isinstance(checks[k], float):
            checks[k] = float(f"{checks[k]:.3g}")
    checks["備考"] = "事後は小数4桁に丸めて出力するため、条件付けの差は丸め誤差（<5e-5）程度"
    res["検算"] = checks
    assert checks["einsum(全同時)との最大差"] < 1e-12 and checks["einsum(祖先集合・周辺)との最大差"] < 1e-12
    assert checks["einsum(条件付け)との最大差"] < 1e-4 and checks["bayesian_inversionとconditionalizationの最大差"] < 1e-3
    return res
