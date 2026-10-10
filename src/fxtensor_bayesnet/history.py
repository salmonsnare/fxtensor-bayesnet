"""日次スナップショットと差分。"""
from __future__ import annotations

import json
import os
import re

from .config import dump_config


def zero(x, nd=1):
    """丸めて -0.0 を 0.0 に正規化する。"""
    x = round(float(x), nd)
    return 0.0 if x == 0 else x


def snapshot_dir(cfg, history_dir):
    return os.path.join(history_dir, cfg["as_of"])


def check_snapshot_free(cfg, history_dir, force=False):
    """スナップショット先が空いているか確認する（何かを書く前に呼ぶ）。"""
    d = snapshot_dir(cfg, history_dir)
    if os.path.exists(d) and not force:
        raise FileExistsError(f"{d} は既にあります（上書きするなら --force）。何も書かずに終了します")
    return d


def resolve_diff_path(path):
    """--diff の引数を (config.json のパス, results.json のパス or None) に解決する。

    フォルダならその中の config.json / results.json を使う。見つからなければ FileNotFoundError。
    """
    if os.path.isdir(path):
        cpath = os.path.join(path, "config.json")
        if not os.path.isfile(cpath):
            raise FileNotFoundError(f"フォルダに config.json がありません: {path}")
    elif os.path.isfile(path):
        cpath = path
    else:
        raise FileNotFoundError(f"比較対象が見つかりません: {path}")
    rpath = os.path.join(os.path.dirname(cpath), "results.json")
    return cpath, (rpath if os.path.isfile(rpath) else None)


def snapshot(cfg, res, history_dir, force=False):
    d = check_snapshot_free(cfg, history_dir, force)
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
    for v in sorted(set(old.get("priors", {})) & set(new.get("priors", {}))):
        if old["priors"][v].get("reason") != new["priors"][v].get("reason"):
            out.append(f"~ 理由文 {v}（事前分布）を更新: {old['priors'][v].get('reason')!r} → {new['priors'][v].get('reason')!r}")
    for v in sorted(set(ov) & set(nv)):
        if ov[v].get("name") != nv[v].get("name"):
            out.append(f"~ 表示名 {v}: {ov[v].get('name')} → {nv[v].get('name')}")
    ok, nk = old.get("kernels", {}), new.get("kernels", {})
    for v in sorted(set(ok) | set(nk)):
        if v not in ok:
            out.append(f"+ 核追加 {v} | {nk[v]['parents']}")
            continue
        if v not in nk:
            out.append(f"- 核削除 {v}")
            continue
        ocpt, ncpt = ok[v].get("cpt", {}), nk[v].get("cpt", {})
        if ok[v]["parents"] != nk[v]["parents"]:
            out.append(f"~ 経路変更 {v}: 親 {ok[v]['parents']} → {nk[v]['parents']}（核の行を全て置き換え）")
            out += [f"  - 旧 {v}[{row}]: {p}" for row, p in ocpt.items()]
            out += [f"  + 新 {v}[{row}]: {p}" for row, p in ncpt.items()]
        else:
            for row in ncpt:
                if row not in ocpt:
                    out.append(f"+ 核の行追加 {v}[{row}]: {ncpt[row]}")
                elif ocpt[row] != ncpt[row]:
                    out.append(f"~ 核 {v}[{row}]: {ocpt[row]} → {ncpt[row]}")
            for row in ocpt:
                if row not in ncpt:
                    out.append(f"- 核の行削除 {v}[{row}]: {ocpt[row]}")
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
            delta = {lab: zero((d[lab] - od.get(lab, 0)) * 100) for lab in d}
            if any(abs(x) >= threshold_pt for x in delta.values()):
                out.append(f"~ {sc} / {v}: " + ", ".join(f"{lab} {x:+.1f}pt" for lab, x in delta.items()))
    return out or ["（結果の変化なし）"]
