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
                errs.append(f"scenarios.{name}: {v} は根の変数ではない")
            else:
                check_dist(f"scenarios.{name}.{v}", p, len(V[v]["labels"]))
    for ob in cfg.get("observations", []):
        if ob.get("var") not in V or ob.get("label") not in V.get(ob.get("var"), {}).get("labels", []):
            errs.append(f"observations: 不正な観測 {ob}")
        for f in ob.get("focus", []):
            if f not in V:
                errs.append(f"observations: focus {f} が未定義")
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
