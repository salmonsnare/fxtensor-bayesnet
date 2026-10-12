"""設定ファイルの読み込み・検証・整形・ハッシュ。"""
from __future__ import annotations

import hashlib
import json
import re
from itertools import product

import numpy as np

TOL = 1e-6


class ConfigError(ValueError):
    """設定ファイルの内容が不正なときに送出される。"""


def load_config(path):
    """JSON 設定を読み込み、検証してから返す。"""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    validate(cfg)
    return cfg


def topo_order(cfg):
    """変数のトポロジカル順（親が先）。循環があれば ConfigError。"""
    vars_ = list(cfg["variables"])
    kernels = cfg.get("kernels", {})
    parents = {v: kernels[v]["parents"] if v in kernels else [] for v in vars_}
    order, seen, temp = [], set(), set()

    def visit(v):
        if v in seen:
            return
        if v in temp:
            raise ConfigError(f"循環があります: {v}")
        temp.add(v)
        for p in parents[v]:
            visit(p)
        temp.discard(v)
        seen.add(v)
        order.append(v)

    for v in vars_:
        visit(v)
    return order


def validate(cfg):
    """設定を検査する。問題があればすべてまとめて ConfigError で報告する。"""
    errs = []
    if "as_of" not in cfg:
        errs.append("as_of（基準日 YYYY-MM-DD）がありません")
    V = cfg.get("variables", {})
    P = cfg.get("priors", {})
    K = cfg.get("kernels", {})
    if not V:
        errs.append("variables が空です")
    for v, spec in V.items():
        labs = spec.get("labels", [])
        if len(labs) < 2 or len(set(labs)) != len(labs):
            errs.append(f"{v}: labels が不正 {labs}")
        if (v in P) == (v in K):
            errs.append(f"{v}: priors（根）か kernels（子）のどちらか一方にだけ定義してください")
    for v in list(P) + list(K):
        if v not in V:
            errs.append(f"{v}: variables に未定義")

    def check_dist(where, p, n):
        if not isinstance(p, list) or len(p) != n:
            errs.append(f"{where}: 長さ {n} のリストが必要（{p}）")
            return
        a = np.asarray(p, dtype=float)
        if (a < -TOL).any():
            errs.append(f"{where}: 負の確率 {p}")
        if abs(a.sum() - 1) > TOL:
            errs.append(f"{where}: 和が1でない（和={a.sum():.6f}）{p}")

    for v, spec in P.items():
        if v in V:
            check_dist(f"priors.{v}", spec.get("p"), len(V[v]["labels"]))
    for v, spec in K.items():
        if v not in V:
            continue
        pars = spec.get("parents", [])
        bad = [p for p in pars if p not in V]
        if bad:
            errs.append(f"kernels.{v}: 未定義の親 {bad}")
            continue
        cpt = spec.get("cpt", {})
        want = {"|".join(c) for c in product(*[V[p]["labels"] for p in pars])}
        have = set(cpt)
        for k in sorted(want - have):
            errs.append(f"kernels.{v}: 行が足りない「{k}」")
        for k in sorted(have - want):
            errs.append(f"kernels.{v}: 余分な行「{k}」（親のラベルと一致しない）")
        for k in sorted(want & have):
            check_dist(f"kernels.{v}[{k}]", cpt[k], len(V[v]["labels"]))
    for name, ov in cfg.get("scenarios", {}).items():
        for v, p in ov.items():
            if v not in P:
                errs.append(f"scenarios.{name}: {v} は根の変数ではない（根でない変数への介入は interventions を使う）")
            else:
                check_dist(f"scenarios.{name}.{v}", p, len(V[v]["labels"]))

    def labels_of(v):
        return V.get(v, {}).get("labels", [])

    for i, ob in enumerate(cfg.get("observations", [])):
        if not isinstance(ob, dict):
            errs.append(f"observations[{i}]: オブジェクトが必要 {ob}")
            continue
        if "evidence" in ob:
            ev = ob["evidence"]
            if "var" in ob or "label" in ob:
                errs.append(f"observations[{i}]: evidence と var/label は同時に書けない")
            if not isinstance(ev, dict) or not ev:
                errs.append(f"observations[{i}]: evidence は空でないオブジェクト {{変数: ラベル or {{\"likelihood\": [...]}}}}")
                ev = {}
            for v, e in ev.items():
                if v not in V:
                    errs.append(f"observations[{i}].evidence: {v} が未定義")
                elif isinstance(e, str):
                    if e not in labels_of(v):
                        errs.append(f"observations[{i}].evidence.{v}: ラベル {e} がない")
                elif isinstance(e, dict) and set(e) == {"likelihood"}:
                    lik = e["likelihood"]
                    n = len(labels_of(v))
                    if not isinstance(lik, list) or len(lik) != n:
                        errs.append(f"observations[{i}].evidence.{v}.likelihood: 長さ {n} のリストが必要（{lik}）")
                    elif any(not isinstance(x, (int, float)) or x < 0 for x in lik) or not any(x > 0 for x in lik):
                        errs.append(f"observations[{i}].evidence.{v}.likelihood: 非負で、少なくとも1つは正の数（{lik}）")
                else:
                    errs.append(f"observations[{i}].evidence.{v}: ラベル文字列か {{\"likelihood\": [...]}} が必要（{e}）")
        elif ob.get("var") not in V or ob.get("label") not in labels_of(ob.get("var")):
            errs.append(f"observations: 不正な観測 {ob}")
        for f in ob.get("focus", []):
            if f not in V:
                errs.append(f"observations: focus {f} が未定義")
    for name, do in (cfg.get("interventions") or {}).items():
        if not isinstance(do, dict) or not do:
            errs.append(f"interventions.{name}: 空でないオブジェクト {{変数: ラベル or 分布}} が必要")
            continue
        for v, spec in do.items():
            if v not in V:
                errs.append(f"interventions.{name}: {v} が未定義")
            elif isinstance(spec, str):
                if spec not in labels_of(v):
                    errs.append(f"interventions.{name}.{v}: ラベル {spec} がない")
            else:
                check_dist(f"interventions.{name}.{v}", spec, len(labels_of(v)))
    effects = cfg.get("effects") or []
    if not isinstance(effects, list):
        errs.append("effects: リストが必要")
        effects = []
    for i, e in enumerate(effects):
        x, y = (e.get("treatment"), e.get("outcome")) if isinstance(e, dict) else (None, None)
        if x not in V or y not in V:
            errs.append(f"effects[{i}]: treatment と outcome は定義済みの変数が必要 {e}")
            continue
        if x == y:
            errs.append(f"effects[{i}]: treatment と outcome が同じ変数")
        for k, var in (("outcome_label", y), ("treated", x), ("control", x)):
            if k in e and e[k] not in labels_of(var):
                errs.append(f"effects[{i}].{k}: {var} にラベル {e[k]} がない")
        if e.get("treated", labels_of(x)[-1]) == e.get("control", labels_of(x)[0]):
            errs.append(f"effects[{i}]: treated と control が同じラベル")
        for gk in ("given_do", "given"):
            g = e.get(gk)
            if g is None:
                continue
            if not isinstance(g, dict) or not g:
                errs.append(f"effects[{i}].{gk}: {{変数: [ラベル, ...]}} の形の空でない辞書が必要")
                continue
            for z, levs in g.items():
                if z not in V:
                    errs.append(f"effects[{i}].{gk}: {z} が未定義")
                    continue
                if z in (x, y):
                    errs.append(f"effects[{i}].{gk}: {z} は treatment / outcome と別の変数が必要")
                if levs is None:
                    continue
                levs = [levs] if isinstance(levs, str) else levs
                if not isinstance(levs, list) or not levs:
                    errs.append(f"effects[{i}].{gk}.{z}: ラベル、ラベルのリスト、または null（全ラベル）が必要")
                    continue
                for lab in levs:
                    if lab not in labels_of(z):
                        errs.append(f"effects[{i}].{gk}.{z}: ラベル {lab} がない")
                if len(set(map(str, levs))) != len(levs):
                    errs.append(f"effects[{i}].{gk}.{z}: ラベルが重複している")
        both = set(e.get("given_do") or {}) & set(e.get("given") or {}) if all(
            isinstance(e.get(k) or {}, dict) for k in ("given_do", "given")) else set()
        if both:
            errs.append(f"effects[{i}]: {', '.join(sorted(both))} が given_do と given の両方にある")
    sens = cfg.get("sensitivity")
    for i, s_ in enumerate(sens if isinstance(sens, list) else [sens] if sens else []):
        if not isinstance(s_, dict) or s_.get("target") not in V:
            errs.append(f"sensitivity[{i}]: target は定義済みの変数が必要 {s_}")
            continue
        if "label" in s_ and s_["label"] not in labels_of(s_["target"]):
            errs.append(f"sensitivity[{i}].label: {s_['target']} にラベル {s_['label']} がない")
        d = s_.get("delta", 0.05)
        if not isinstance(d, (int, float)) or not 0 < d < 1:
            errs.append(f"sensitivity[{i}].delta: 0 より大きく 1 より小さい数（{d}）")
        tp = s_.get("top", 10)
        if not isinstance(tp, int) or tp < 1:
            errs.append(f"sensitivity[{i}].top: 1 以上の整数（{tp}）")
    for o in cfg.get("outputs", []):
        if o not in V:
            errs.append(f"outputs: {o} が未定義")
    if not errs:
        try:
            topo_order(cfg)
        except ConfigError as e:
            errs.append(str(e))
    if errs:
        raise ConfigError("設定エラー:\n  - " + "\n  - ".join(errs))


def config_hash(cfg):
    """設定内容の短いハッシュ（キー順に依存しない）。"""
    return hashlib.sha256(json.dumps(cfg, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def dump_config(cfg):
    """核の1行・ラベル列を1行にまとめた読みやすい JSON（差分が行単位で読める）。"""
    s = json.dumps(cfg, ensure_ascii=False, indent=2)
    s = re.sub(r"\[\s*((?:-?[0-9.eE+-]+\s*,\s*)*-?[0-9.eE+-]+)\s*\]",
               lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", s)
    s = re.sub(r'\[\s*((?:"[^"\n]*"\s*,\s*)*"[^"\n]*")\s*\]',
               lambda m: "[" + ", ".join(x.strip() for x in re.findall(r'"[^"\n]*"', m.group(1))) + "]", s)
    return s + "\n"
