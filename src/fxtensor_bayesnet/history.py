"""日次スナップショットと差分。"""
from __future__ import annotations

import json
import os
import re

from .config import dump_config


def snapshot(cfg, res, history_dir, force=False):
    d = os.path.join(history_dir, cfg["as_of"])
    if os.path.exists(d) and not force:
        raise FileExistsError(f"{d} は既にあります（上書きするなら --force）")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        f.write(dump_config(cfg))
    with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return d


def previous_snapshot(history_dir, as_of):
    """history_dir のうち as_of より前で最新の日付の config.json（なければ None）。"""
    if not os.path.isdir(history_dir):
        return None
    dates = sorted(d for d in os.listdir(history_dir)
                   if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) and d < as_of
                   and os.path.exists(os.path.join(history_dir, d, "config.json")))
    return os.path.join(history_dir, dates[-1], "config.json") if dates else None


def diff_configs(old, new):
    out = []
    ov, nv = old.get("variables", {}), new.get("variables", {})
    for v in sorted(set(ov) | set(nv)):
        if v not in ov:
            out.append(f"+ 変数追加 {v}: {nv[v]['labels']}")
        elif v not in nv:
            out.append(f"- 変数削除 {v}")
        elif ov[v]["labels"] != nv[v]["labels"]:
            out.append(f"~ ラベル変更 {v}: {ov[v]['labels']} → {nv[v]['labels']}")
    for v in sorted(set(old.get("priors", {})) | set(new.get("priors", {}))):
        a = old.get("priors", {}).get(v, {}).get("p")
        b = new.get("priors", {}).get(v, {}).get("p")
        if a != b:
            out.append(f"~ 事前分布 {v}: {a} → {b}")
    ok, nk = old.get("kernels", {}), new.get("kernels", {})
    for v in sorted(set(ok) | set(nk)):
        if v not in ok:
            out.append(f"+ 核追加 {v} | {nk[v]['parents']}")
            continue
        if v not in nk:
            out.append(f"- 核削除 {v}")
            continue
        if ok[v]["parents"] != nk[v]["parents"]:
            out.append(f"~ 経路変更 {v}: 親 {ok[v]['parents']} → {nk[v]['parents']}")
            continue
        for row in nk[v]["cpt"]:
            if ok[v]["cpt"].get(row) != nk[v]["cpt"][row]:
                out.append(f"~ 核 {v}[{row}]: {ok[v]['cpt'].get(row)} → {nk[v]['cpt'][row]}")
        if ok[v].get("reason") != nk[v].get("reason"):
            out.append(f"~ 理由文 {v} を更新")
    os_, ns = old.get("scenarios", {}), new.get("scenarios", {})
    for s in sorted(set(os_) | set(ns)):
        if os_.get(s) != ns.get(s):
            out.append(f"~ シナリオ {s}: {os_.get(s)} → {ns.get(s)}")
    if old.get("observations") != new.get("observations"):
        out.append("~ observations を変更")
    if old.get("outputs") != new.get("outputs"):
        out.append(f"~ outputs: {old.get('outputs')} → {new.get('outputs')}")
    return out or ["（設定の変更なし）"]


def diff_results(old, new, threshold_pt=0.1):
    out = []
    for sc, r in new.get("シナリオ", {}).items():
        o = old.get("シナリオ", {}).get(sc)
        if not o:
            out.append(f"+ シナリオ {sc}（前回なし）")
            continue
        for v, d in r["周辺"].items():
            od = o["周辺"].get(v)
            if od is None:
                out.append(f"+ {sc} / {v}（前回なし）")
                continue
            delta = {lab: round((d[lab] - od.get(lab, 0)) * 100, 1) for lab in d}
            if any(abs(x) >= threshold_pt for x in delta.values()):
                out.append(f"~ {sc} / {v}: " + ", ".join(f"{lab} {x:+.1f}pt" for lab, x in delta.items()))
    return out or ["（結果の変化なし）"]
