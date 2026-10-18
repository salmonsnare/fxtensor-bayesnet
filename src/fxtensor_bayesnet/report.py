"""出力：Markdown の結果表と mermaid の構造図。"""
from __future__ import annotations

import hashlib
import json
import re

from .i18n import obs_text, scenario_name, t


def render_markdown(res, cfg, lang="ja"):
    """結果の Markdown を返す。lang は ja または en（見出しの言語。結果辞書は日本語キーのまま読む）。"""
    V = cfg["variables"]
    hz = f"{t(lang, 'horizon_sep')}{res['ホライズン']}" if res.get("ホライズン") else ""
    L = [t(lang, "result_title", as_of=res["基準日"], horizon=hz), ""]
    if res.get("注意"):
        L += [f"> {res['注意']}", ""]
    outputs = cfg.get("outputs") or list(next(iter(res["シナリオ"].values()))["周辺"])
    for v in outputs:
        labs = V[v]["labels"]
        name = V[v].get("name", v)
        L += [f"## {v}（{name}）" if lang == "ja" else f"## {v} ({name})",
              f"| {t(lang, 'scenario')} | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        for sc, r in res["シナリオ"].items():
            L.append(f"| {scenario_name(lang, sc)} | " + " | ".join(f"{x*100:.1f}%" for x in r["周辺"][v].values()) + " |")
        L.append("")
    L.append(t(lang, "bayes_header"))
    for k, r in res["ベイズ更新(基本)"].items():
        L.append(obs_text(lang, "obs_header", k, r))
        if r.get("不可能な観測"):
            L += [f"- {t(lang, 'impossible')}", ""]
            continue
        for w, d in r["事後"].items():
            L.append(f"- {w}: " + " / ".join(f"{a} {b*100:.1f}%" for a, b in d.items()))
        L.append(f"- {t(lang, 'ranking')}: {' > '.join(r['動きの大きさ順(TV)'])}")
        for w, mv in r["注目変数の動き"].items():
            L.append(f"  - {w}: TV {mv['全変動距離TV']}, KL {mv['KL(事後||事前)[nat]']}, {t(lang, 'delta')} {mv['差(pt)']}")
        L.append("")
    L += _md_interventions(res, cfg, lang, outputs)
    L += _md_effects(res, cfg, lang)
    L += _md_sensitivity(res, lang)
    L.append(f"{t(lang, 'checks')}: {json.dumps(res['検算'], ensure_ascii=False)}")
    L += _md_unchecked(res["検算"], lang)
    return "\n".join(L) + "\n"


def _pt(x, lang):
    return t(lang, "na") if x is None else f"{x * 100:+.1f}pt"


def _pct(x, lang):
    return t(lang, "na") if x is None else f"{x * 100:.1f}%"


def _md_interventions(res, cfg, lang, outputs):
    if not res.get("介入"):
        return []
    V = cfg["variables"]
    L = [t(lang, "interventions_header"), ""]
    for v in outputs:
        labs = V[v]["labels"]
        L += [f"### {v}（{V[v].get('name', v)}）" if lang == "ja" else f"### {v} ({V[v].get('name', v)})",
              f"| {t(lang, 'intervention')} | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        base = res["シナリオ"]["基本"]["周辺"].get(v)
        if base:
            L.append(f"| {scenario_name(lang, '基本')} | " + " | ".join(f"{base[lab]*100:.1f}%" for lab in labs) + " |")
        for name, r in res["介入"].items():
            L.append(f"| {name} | " + " | ".join(f"{r['周辺'][v][lab]*100:.1f}%" for lab in labs) + " |")
        L.append("")
    return L


def _md_unchecked(checks, lang):
    """未検算の項目（最大差が null）と、検算を省略した観測を明示する（「検算して差 0」と区別するため）。"""
    from .keys import KEY_MAP
    L = []
    none = [k for k, v in checks.items() if k.endswith("最大差") and v is None]
    if none:
        names = none if lang == "ja" else [KEY_MAP.get(k, k) for k in none]
        L.append("- " + t(lang, "not_checked", keys=", ".join(f"`{k}`" for k in names)))
    for k, kind in (("条件付けの検算を省略した観測", "einsum"), ("bayesian_inversionの検算を省略した観測", "bayesian_inversion")):
        if checks.get(k):
            L.append("- " + t(lang, "skipped_obs", kind=kind, obs=", ".join(checks[k])))
    return [""] + L if L else []


def _md_effects(res, cfg, lang):
    if not res.get("因果効果"):
        return []
    L = [t(lang, "effects_header"), ""]
    for key, e in res["因果効果"].items():
        x, y = e["処置"], f"{e['結果']}={e['結果のラベル']}"
        L += [f"### {key}", f"| {t(lang, 'level')} | {t(lang, 'observed_col', x=x, y=y)} | {t(lang, 'do_col', x=x, y=y)} |",
              "|---|---|---|"]
        for lev, p in e["P(結果|do(処置))"].items():
            L.append(f"| {lev} | {_pct(e['P(結果|処置)(観測)'][lev], lang)} | {_pct(p, lang)} |")
        L += ["", "- " + t(lang, "effect_line", t=e["処置群"], c=e["対照群"], od=_pt(e["観測上の差"], lang),
                             ace=_pt(e["ACE"], lang), gap=_pt(e["交絡による差(観測上の差−ACE)"], lang))]
        if e.get("観測確率0の水準"):
            L.append("- " + t(lang, "zero_level_md", levs=", ".join(e["観測確率0の水準"])))
        L.append("")
        for sk, hk, pk, ck in (("層別ACE(given_do)", "strat_do_header", "P(結果|do(処置),do(層))", "strat_do_col"),
                               ("条件付きACE(given)", "strat_obs_header", "P(結果|do(処置),層)", "strat_obs_col")):
            s = e.get(sk)
            if not s:
                continue
            levs = list(e["P(結果|do(処置))"])
            L += [t(lang, hk, z=", ".join(s["条件"])), "",
                  f"| {t(lang, 'stratum')} | " + " | ".join(t(lang, ck, x=x, y=y, lev=lev) for lev in levs) + " | ACE |",
                  "|---|" + "---|" * (len(levs) + 1)]
            for name, layer in s["層"].items():
                L.append(f"| {name} | " + " | ".join(_pct(layer[pk][lev], lang) for lev in levs)
                         + f" | {_pt(layer['ACE'], lang)} |")
            L.append("")
            names = list(s["層"])
            if len(names) >= 2:
                L.append("- " + t(lang, "strat_diff", last=names[-1], first=names[0],
                                  d=_pt(s["層間のACEの差(最後−最初)"], lang)))
            if any(layer["ACE"] is None for layer in s["層"].values()):
                L.append("- " + t(lang, "strat_undefined", x=x))
            if L[-1] != "":
                L.append("")
    return L


def _md_sensitivity(res, lang):
    L = []
    for key, s in (res.get("感度分析") or {}).items():
        L += [t(lang, "sens_header", y=key, p=s["基準のP"] * 100, d=s["変化量delta"], n=s["パラメータ数"], k=len(s["上位"])),
              "", t(lang, "sens_cols"), "|---|---|---|---|---|"]
        for r in s["上位"]:
            name = r["パラメータ"].replace("|", "\\|")        # 表の区切りと区別する
            L.append(f"| {name} | {r['基準値']:.3g} → {r['下げた値']:.3g} / {r['上げた値']:.3g} | "
                     f"{r['P(下げたとき)']*100:.1f}% | {r['P(上げたとき)']*100:.1f}% | {r['幅(pt)']:.2f} |")
        L.append("")
    return L


def render_mermaid(cfg):
    """ネットワーク構造の mermaid（``graph TD``）。根・中間・出力を色分けする。"""
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
    """mermaid のノード id。英数字と ``_`` 以外は短いハッシュに置き換える。"""
    return v if re.fullmatch(r"[A-Za-z0-9_]+", v) else "n_" + hashlib.md5(v.encode()).hexdigest()[:6]
