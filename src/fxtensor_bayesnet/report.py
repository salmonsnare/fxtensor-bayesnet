"""出力：Markdown の結果表と mermaid の構造図。"""
from __future__ import annotations

import hashlib
import json
import re


def render_markdown(res, cfg):
    V = cfg["variables"]
    title = f"# 結果（基準日 {res['基準日']}"
    if res.get("ホライズン"):
        title += f"、{res['ホライズン']}"
    L = [title + "）", ""]
    if res.get("注意"):
        L += [f"> {res['注意']}", ""]
    outputs = cfg.get("outputs") or list(next(iter(res["シナリオ"].values()))["周辺"])
    for v in outputs:
        labs = V[v]["labels"]
        L += [f"## {v}（{V[v].get('name', v)}）", "| シナリオ | " + " | ".join(labs) + " |",
              "|---|" + "---|" * len(labs)]
        for name, r in res["シナリオ"].items():
            L.append(f"| {name} | " + " | ".join(f"{x*100:.1f}%" for x in r["周辺"][v].values()) + " |")
        L.append("")
    L.append("## ベイズ更新（基本）")
    for k, r in res["ベイズ更新(基本)"].items():
        L.append(f"### {k}（観測前の確率 {r['観測前の確率']*100:.1f}%）")
        for w, d in r["事後"].items():
            L.append(f"- {w}: " + " / ".join(f"{a} {b*100:.1f}%" for a, b in d.items()))
        L.append(f"- 動きの大きさ順(TV): {' > '.join(r['動きの大きさ順(TV)'])}")
        for w, mv in r["注目変数の動き"].items():
            L.append(f"  - {w}: TV {mv['全変動距離TV']}, KL {mv['KL(事後||事前)[nat]']}, 差 {mv['差(pt)']}")
        L.append("")
    L.append(f"検算: {json.dumps(res['検算'], ensure_ascii=False)}")
    return "\n".join(L) + "\n"


def render_mermaid(cfg):
    V = cfg["variables"]

    def node(v):
        return f'{safe(v)}["{v}: {V[v].get("name", v)}<br/>{"/".join(V[v]["labels"])}"]'

    L = ["graph TD"]
    for v in V:
        L.append("  " + node(v))
    for v, spec in cfg["kernels"].items():
        for p in spec["parents"]:
            L.append(f"  {safe(p)} --> {safe(v)}")
    # 根・中間・出力の色分け（図の PNG と同じ配色）
    outputs = set(cfg.get("outputs") or [v for v in V if v not in cfg["priors"]])
    kinds = {"root": [v for v in V if v in cfg["priors"]],
             "middle": [v for v in V if v not in cfg["priors"] and v not in outputs],
             "output": [v for v in V if v not in cfg["priors"] and v in outputs]}
    styles = {"root": "fill:#cfe3f7,stroke:#3b78b5", "middle": "fill:#e6e6e6,stroke:#777777",
              "output": "fill:#fde2c4,stroke:#d9822b"}
    for k, vs in kinds.items():
        if vs:
            L.append(f"  classDef {k} {styles[k]}")
            L.append(f"  class {','.join(safe(v) for v in vs)} {k}")
    return "\n".join(L) + "\n"


def safe(v):
    return v if re.fullmatch(r"[A-Za-z0-9_]+", v) else "n_" + hashlib.md5(v.encode()).hexdigest()[:6]
