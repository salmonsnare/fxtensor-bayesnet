"""介入（do 演算・グラフ手術）、平均因果効果 ACE、CPT の感度分析。"""
from __future__ import annotations

import copy
from itertools import product

import numpy as np


def do_vector(labels, spec):
    """介入の指定（ラベル文字列＝ハードな do、確率のリスト＝確率的な介入）を分布ベクトルにする。"""
    if isinstance(spec, str):
        return np.array([1.0 if lab == spec else 0.0 for lab in labels])
    return np.asarray(spec, float)


def do_config(cfg, do):
    """グラフ手術：do = {変数: ラベル or 分布} の各変数について、親からの辺（核）を外し、
    指定の分布を事前分布として置いた派生設定を返す。元の設定は変更しない。

    派生設定では scenarios / observations / interventions / effects / sensitivity は空にする。
    """
    c = copy.deepcopy(cfg)
    for v, spec in do.items():
        labs = c["variables"][v]["labels"]
        c.setdefault("kernels", {}).pop(v, None)
        c["priors"][v] = {"p": [float(x) for x in do_vector(labs, spec)], "reason": f"do({v}={spec})"}
    for k in ("scenarios", "interventions"):
        c[k] = {}
    for k in ("observations", "effects", "sensitivity"):
        c.pop(k, None)
    return c


def given_levels(labels, spec):
    """given_do / given の 1 変数分（None＝全ラベル、ラベル文字列、ラベルのリスト）をラベルのリストにする。"""
    if spec is None:
        return list(labels)
    if isinstance(spec, str):
        return [spec]
    return list(spec)


def strata(given):
    """{変数: [ラベル, ...]} から層（ラベルの組み合わせ）のリスト [(層の名前, {変数: ラベル}), ...] を作る。
    層の名前は「Z=z」（複数の変数なら「Z=z, W=w」）。"""
    vs = list(given)
    out = []
    for combo in product(*[given[v] for v in vs]):
        d = dict(zip(vs, combo))
        out.append((", ".join(f"{v}={lab}" for v, lab in d.items()), d))
    return out


def effect_spec(cfg, e):
    """effects の 1 項目に既定値を補う（outcome_label=最後のラベル、treated=最後、control=最初）。
    given_do / given（層別・条件付き ACE）は {変数: [ラベル, ...]} に正規化し、指定がなければ空の辞書。"""
    V = cfg["variables"]
    tl, ol = V[e["treatment"]]["labels"], V[e["outcome"]]["labels"]
    return {"treatment": e["treatment"], "outcome": e["outcome"],
            "outcome_label": e.get("outcome_label", ol[-1]),
            "treated": e.get("treated", tl[-1]), "control": e.get("control", tl[0]),
            "given_do": {v: given_levels(V[v]["labels"], s) for v, s in (e.get("given_do") or {}).items()},
            "given": {v: given_levels(V[v]["labels"], s) for v, s in (e.get("given") or {}).items()}}


def sensitivity_specs(cfg):
    """sensitivity（1 つのオブジェクトまたはリスト）をリストにし、既定値を補う。"""
    s = cfg.get("sensitivity")
    if not s:
        return []
    items = s if isinstance(s, list) else [s]
    V = cfg["variables"]
    return [{"target": x["target"], "label": x.get("label", V[x["target"]]["labels"][-1]),
             "delta": float(x.get("delta", 0.05)), "top": int(x.get("top", 10))} for x in items]


def perturb_row(p, j, new_pj):
    """行 p の j 番目を new_pj にし、残りを比例的に縮めて和を 1 に保つ（比例共変）。"""
    p = np.asarray(p, float)
    q = p.copy()
    rest = 1.0 - p[j]
    if rest > 1e-15:
        q *= (1.0 - new_pj) / rest
    else:  # 残りが全部 0 なら等分
        q[:] = (1.0 - new_pj) / (len(p) - 1)
    q[j] = new_pj
    return q


def sensitivity(net, priors, target, label, delta=0.05, top=10):
    """P(target=label) の、各パラメータ（根の事前分布・祖先の CPT の各行の各要素）への一変数感度。

    各パラメータを ±delta 動かし（[0,1] に切り詰め）、同じ行の残りは比例的に変える。
    2 値の行は 1 要素だけ（最後のラベル）を動かす（もう一方は同じ変化の裏返しなので）。
    戻り値: (基準の確率, パラメータ数, 幅の大きい順の上位 top 件のリスト)
    """
    yi = net.labels[target].index(label)
    base = float(net.contract(priors, [target])[yi])
    rows = []
    for v in net.ancestors([target]):
        labs = net.labels[v]
        js = [len(labs) - 1] if len(labs) == 2 else range(len(labs))
        if v in net.parents:
            arr = net.cpt[v]
            pars = net.parents[v]
            for idx in product(*[range(len(net.labels[p])) for p in pars]):
                rows.append((v, f"kernels.{v}[{'|'.join(net.labels[p][i] for p, i in zip(pars, idx))}]", arr, idx, js))
        else:
            rows.append((v, f"priors.{v}", np.asarray(priors[v], float), (), js))
    out = []
    for v, name, arr, idx, js in rows:
        p = arr[idx] if idx else arr
        for j in js:
            vals = {}
            for sgn in (-1, +1):
                nj = float(np.clip(p[j] + sgn * delta, 0.0, 1.0))
                a2 = np.array(arr, float, copy=True)
                if idx:
                    a2[idx] = perturb_row(p, j, nj)
                else:
                    a2 = perturb_row(p, j, nj)
                vals[sgn] = (nj, float(net.contract(priors, [target], cpt={v: a2})[yi]))
            out.append({"パラメータ": f"{name} → {net.labels[v][j]}", "基準値": float(p[j]),
                        "下げた値": vals[-1][0], "上げた値": vals[+1][0],
                        "P(下げたとき)": vals[-1][1], "P(上げたとき)": vals[+1][1],
                        "幅(pt)": abs(vals[+1][1] - vals[-1][1]) * 100})
    out.sort(key=lambda r: -r["幅(pt)"])
    return base, len(out), out[:top]
