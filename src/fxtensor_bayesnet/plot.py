"""図の出力（matplotlib。optional 依存：pip install 'fxtensor-bayesnet[plot]'）。

- plot_network   … ネットワーク構造図（階層レイアウト、根・中間・出力を色分け）
- plot_marginals … 出力変数ごとの周辺分布の棒グラフ（シナリオ比較）
- plot_updates   … 観測ごとの注目変数の事前・事後比較
- plot_diff      … 前回スナップショットからの周辺分布の変化量（pt）
"""
from __future__ import annotations

import os
import warnings

FONT_CANDIDATES = [
    "Noto Sans CJK JP", "Noto Sans JP", "Source Han Sans JP", "IPAexGothic", "IPAGothic",
    "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", "MS Gothic",
    "TakaoGothic", "VL Gothic", "Noto Sans CJK SC", "Noto Sans Mono CJK JP", "Noto Serif CJK JP",
]
FONT_ENV = "FXTENSOR_BAYESNET_FONT"   # 使うフォント名を明示したいとき

COLORS = {"root": "#cfe3f7", "middle": "#e6e6e6", "output": "#fde2c4"}
EDGE_COLORS = {"root": "#3b78b5", "middle": "#777777", "output": "#d9822b"}
PRIOR_COLOR, POST_COLOR = "#9fb4c7", "#d9822b"


class PlotError(RuntimeError):
    """図を描けないとき（matplotlib がない等）。"""


_plt = None
_font = False


def _mpl():
    """matplotlib を遅延読み込み（Agg バックエンド）。なければ分かりやすい PlotError。"""
    global _plt
    if _plt is None:
        try:
            import matplotlib
        except ImportError as e:
            raise PlotError("図の出力には matplotlib が必要です。"
                            "pip install 'fxtensor-bayesnet[plot]'（または pip install matplotlib）で入れてください。") from e
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
        setup_font()
    return _plt


def setup_font():
    """日本語を表示できるフォントを自動検出して設定する。見つからなければ警告し既定フォントのまま。"""
    global _font
    if _font is not False:
        return _font
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    cands = ([os.environ[FONT_ENV]] if os.environ.get(FONT_ENV) else []) + FONT_CANDIDATES
    found = next((c for c in cands if c in names), None)
    if found:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [found] + list(matplotlib.rcParams["font.sans-serif"])
    else:
        warnings.warn("日本語フォントが見つかりません（Noto Sans CJK JP など）。既定フォントで描くため、"
                      f"日本語が表示されない場合があります。環境変数 {FONT_ENV} でフォント名を指定できます。",
                      RuntimeWarning, stacklevel=2)
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["svg.fonttype"] = "path"     # SVG を閲覧環境のフォントに依存させない
    _font = found
    return found


def _save(fig, path, dpi):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    _mpl().close(fig)
    return path


# ---------------------------------------------------------------------------
# (1) 構造図
# ---------------------------------------------------------------------------
def node_kinds(cfg):
    outputs = set(cfg.get("outputs") or [v for v in cfg["variables"] if v not in cfg["priors"]])
    return {v: "root" if v in cfg["priors"] else "output" if v in outputs else "middle"
            for v in cfg["variables"]}


def layered_layout(cfg):
    """階層レイアウト（Sugiyama 法の簡易版）。

    層 = 根からの最長経路長。2 層以上をまたぐ辺には各層にダミー点を置き、
    層内の並びは重心法（上から・下から数回）で交差を減らす。
    戻り値: layers（各層の要素のリスト。ダミーは ("__d", 親, 子, 層) のタプル）、level、parents、
    chains（(親, 子) → 経由するダミーのリスト）
    """
    from .config import topo_order
    order = topo_order(cfg)
    parents = {v: cfg["kernels"][v]["parents"] if v in cfg["kernels"] else [] for v in order}
    level = {}
    for v in order:
        level[v] = 1 + max((level[p] for p in parents[v]), default=-1)
    nl = max(level.values()) + 1
    layers = [[v for v in order if level[v] == i] for i in range(nl)]
    up = {v: [] for v in order}      # 1 層上の隣接（ダミー含む）
    down = {v: [] for v in order}
    chains = {}
    for v in order:
        for p in parents[v]:
            prev, chain = p, []
            for lv in range(level[p] + 1, level[v]):
                d = ("__d", p, v, lv)
                layers[lv].append(d)
                up[d], down[d] = [prev], []
                down[prev].append(d)
                chain.append(d)
                prev = d
            up[v].append(prev)
            down[prev].append(v)
            chains[(p, v)] = chain

    def place(layer):
        return {x: j - (len(layer) - 1) / 2 for j, x in enumerate(layer)}

    pos = {}
    for layer in layers:
        pos.update(place(layer))
    for _ in range(4):
        for i in list(range(1, nl)) + list(range(nl - 2, -1, -1)):
            nb = up if i > 0 else down
            def bary(x, nb=nb):
                ns = nb[x] or (down[x] if nb is up else up[x])
                return sum(pos[n] for n in ns) / len(ns) if ns else pos[x]
            layers[i].sort(key=bary)
            pos.update(place(layers[i]))
        layers[0].sort(key=lambda x: (sum(pos[n] for n in down[x]) / len(down[x])) if down[x] else pos[x])
        pos.update(place(layers[0]))
    return layers, level, parents, chains


def _text_width(s, size_pt, bold=False):
    """おおよその文字列幅（インチ）。全角 1em、半角 0.6em。"""
    em = sum(1.0 if ord(ch) > 0x2E7F else 0.6 for ch in s)
    return em * size_pt / 72 * (1.08 if bold else 1.0)


def plot_network(cfg, path, dpi=150, title=None):
    plt = _mpl()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
    from matplotlib.path import Path
    V = cfg["variables"]
    kinds = node_kinds(cfg)
    layers, level, parents, chains = layered_layout(cfg)
    fs_name, fs_lab = 11, 8.5
    texts = {v: (V[v].get("name", v), f"{v}  ［{' / '.join(V[v]['labels'])}］") for v in V}
    widths = {v: max(_text_width(texts[v][0], fs_name, True), _text_width(texts[v][1], fs_lab)) + 0.4 for v in V}
    box_h, gap_x, gap_y, dummy_w = 0.72, 0.45, 0.85, 0.35

    def w_of(x):
        return dummy_w if isinstance(x, tuple) else widths[x]

    row_w = [sum(w_of(x) for x in layer) + gap_x * (len(layer) - 1) for layer in layers]
    W = max(row_w)
    xy = {}
    for i, layer in enumerate(layers):
        # 層の幅が全体より狭いときは間隔を広げて中央寄せ
        extra = (W - row_w[i]) / (len(layer) + 1) if len(layer) > 1 else 0.0
        x = -(row_w[i] + extra * (len(layer) - 1)) / 2
        for item in layer:
            w = w_of(item)
            xy[item] = (x + w / 2, -(i * (box_h + gap_y)))
            x += w + gap_x + extra
    ys = [y for _, y in xy.values()]
    y_top, y_bot = max(ys) + box_h / 2 + 0.1, min(ys) - box_h / 2 - 0.1
    fig, ax = plt.subplots(figsize=(W + 0.4, y_top - y_bot + 0.2))
    ax.set_xlim(-W / 2 - 0.1, W / 2 + 0.1)
    ax.set_ylim(y_bot, y_top)
    ax.set_aspect("equal")
    ax.axis("off")
    # 箱の上下の辺に、出入りする辺の接続点を相手の位置順に散らす
    edges = [(p, v) for v, ps in parents.items() for p in ps]
    nxt = {e: (xy[chains[e][0]][0] if chains[e] else xy[e[1]][0]) for e in edges}
    prv = {e: (xy[chains[e][-1]][0] if chains[e] else xy[e[0]][0]) for e in edges}
    port_out, port_in = {}, {}
    for v in V:
        for es, key, store in (([e for e in edges if e[0] == v], nxt, port_out),
                               ([e for e in edges if e[1] == v], prv, port_in)):
            es.sort(key=lambda e: key[e])
            step = min(0.3, 0.7 * widths[v] / max(len(es), 1))
            for k, e in enumerate(es):
                store[e] = xy[v][0] + (k - (len(es) - 1) / 2) * step
    for (p, v) in edges:
        pts = [(port_out[(p, v)], xy[p][1] - box_h / 2)]
        for d in chains[(p, v)]:          # ダミー点では層の上端と下端を通る（箱の間を縦に抜ける）
            x, y = xy[d]
            pts += [(x, y + box_h / 2), (x, y - box_h / 2)]
        pts.append((port_in[(p, v)], xy[v][1] + box_h / 2))
        # 縦向きの接線を持つ 3 次ベジェで滑らかにつなぐ
        verts, codes = [pts[0]], [Path.MOVETO]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            h = (y0 - y1) * 0.5
            verts += [(x0, y0 - h), (x1, y1 + h), (x1, y1)]
            codes += [Path.CURVE4] * 3
        ax.add_patch(FancyArrowPatch(path=Path(verts, codes), arrowstyle="-|>", mutation_scale=14,
                                     lw=1.2, color="#555555", shrinkA=0, shrinkB=2, zorder=1))
    for v in V:
        x, y = xy[v]
        w, k = widths[v], kinds[v]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - box_h / 2), w, box_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=COLORS[k], ec=EDGE_COLORS[k], lw=1.5, zorder=2))
        ax.text(x, y + 0.12, texts[v][0], ha="center", va="center", fontsize=fs_name, weight="bold", zorder=3)
        ax.text(x, y - 0.16, texts[v][1], ha="center", va="center", fontsize=fs_lab, color="#333333", zorder=3)
    handles = [Patch(fc=COLORS[k], ec=EDGE_COLORS[k], label=lab)
               for k, lab in (("root", "根（事前分布）"), ("middle", "中間変数"), ("output", "出力変数"))
               if k in kinds.values()]
    ax.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, 0.0))
    ax.set_title(title or "ネットワーク構造", fontsize=12)
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# (2) 周辺分布（シナリオ比較）
# ---------------------------------------------------------------------------
def _grid(n, ncols):
    ncols = max(1, min(ncols, n))
    return (n + ncols - 1) // ncols, ncols


def _grouped_bars(ax, labels, series, colors, annotate=True):
    import numpy as np
    k = len(series)
    width = 0.8 / k
    x = np.arange(len(labels))
    for i, (name, vals) in enumerate(series):
        xs = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(xs, [v * 100 for v in vals], width * 0.92, label=name, color=colors[i % len(colors)])
        if annotate:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{v*100:.0f}",
                        ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("%", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_marginals(res, cfg, path, dpi=150, ncols=2):
    plt = _mpl()
    V = cfg["variables"]
    scen = list(res["シナリオ"])
    outputs = list(next(iter(res["シナリオ"].values()))["周辺"])
    nr, nc = _grid(len(outputs), ncols)
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.1 * nr), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    for ax, v in zip(axes.flat, outputs):
        labs = V[v]["labels"]
        series = [(s, [res["シナリオ"][s]["周辺"][v][lab] for lab in labs]) for s in scen]
        _grouped_bars(ax, labs, series, colors, annotate=len(scen) * len(labs) <= 15)
        ax.set_title(f"{V[v].get('name', v)}（{v}）", fontsize=10)
    for ax in list(axes.flat)[len(outputs):]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(scen), 4), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), title="シナリオ", title_fontsize=9)
    fig.suptitle(f"周辺分布のシナリオ比較（基準日 {res['基準日']}）", fontsize=12)
    fig.tight_layout(rect=(0, 0.06 + 0.02 * ((len(scen) - 1) // 4), 1, 0.97))
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# (3) ベイズ更新（事前・事後）
# ---------------------------------------------------------------------------
def plot_updates(res, cfg, path, dpi=150):
    plt = _mpl()
    V = cfg["variables"]
    ups = res["ベイズ更新(基本)"]
    if not ups:
        return None
    rows = []
    for key, r in ups.items():
        focus = [w for w in r["注目変数の事前"] if w in r["事後"]]
        rows.append((key, r, focus))
    nc = max(1, max(len(f) for _, _, f in rows))
    fig, axes = plt.subplots(len(rows), nc, figsize=(3.5 * nc, 2.9 * len(rows)), squeeze=False)
    for i, (key, r, focus) in enumerate(rows):
        for j in range(nc):
            ax = axes[i][j]
            if j >= len(focus):
                ax.axis("off")
                continue
            w = focus[j]
            labs = V[w]["labels"]
            pri = [r["注目変数の事前"][w][lab] for lab in labs]
            post = [r["事後"][w][lab] for lab in labs]
            _grouped_bars(ax, labs, [("事前", pri), ("事後", post)], [PRIOR_COLOR, POST_COLOR])
            tv = r["注目変数の動き"].get(w, {}).get("全変動距離TV")
            ax.set_title(f"{V[w].get('name', w)}" + (f"（TV {tv:.3f}）" if tv is not None else ""), fontsize=9.5)
        obs = key.replace(" を観測", "")
        axes[i][0].annotate(f"観測: {obs}（観測前の確率 {r['観測前の確率']*100:.1f}%）",
                            xy=(0, 1.2), xycoords="axes fraction", fontsize=10.5, weight="bold", ha="left")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("ベイズ更新：注目変数の事前・事後", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# (4) --diff の変化量
# ---------------------------------------------------------------------------
def diff_deltas(old_res, new_res):
    """{シナリオ: [(変数, ラベル, 変化pt)]}（両方にあるシナリオ・変数のみ）。"""
    out = {}
    for sc, r in new_res.get("シナリオ", {}).items():
        o = old_res.get("シナリオ", {}).get(sc)
        if not o:
            continue
        rows = []
        for v, d in r["周辺"].items():
            od = o["周辺"].get(v)
            if od is None:
                continue
            rows += [(v, lab, (d[lab] - od.get(lab, 0)) * 100) for lab in d]
        out[sc] = rows
    return out


def plot_diff(old_res, new_res, cfg, path, dpi=150, old_label="前回", new_label="今回"):
    plt = _mpl()
    V = cfg["variables"]
    deltas = diff_deltas(old_res, new_res)
    if not deltas:
        return None
    n = len(deltas)
    nrows = max(len(r) for r in deltas.values())
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 1.6, 0.27 * nrows + 1.6), squeeze=False, sharey=True)
    lim = max([abs(x) for rows in deltas.values() for _, _, x in rows] + [0.5]) * 1.25
    for ax, (sc, rows) in zip(axes[0], deltas.items()):
        ys = list(range(len(rows)))[::-1]
        vals = [x for _, _, x in rows]
        ax.barh(ys, vals, color=["#d9822b" if x > 0 else "#3b78b5" for x in vals], height=0.7)
        for y, x in zip(ys, vals):
            if abs(x) >= 0.05:
                ax.text(x + (0.02 * lim if x >= 0 else -0.02 * lim), y, f"{x:+.1f}", va="center",
                        ha="left" if x >= 0 else "right", fontsize=7)
        if all(abs(x) < 0.05 for x in vals):
            ax.text(0.5, 0.5, "変化なし", transform=ax.transAxes, ha="center", va="center", fontsize=11,
                    color="#666666", bbox=dict(fc="white", ec="none"))
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_title(sc, fontsize=10)
        ax.set_xlabel("変化（pt）", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{V.get(v, {}).get('name', v)}: {lab}" for v, lab, _ in rows], fontsize=8)
        # 変数の境目に薄い線
        prev = None
        for y, (v, _, _) in zip(ys, rows):
            if prev is not None and v != prev:
                ax.axhline(y + 0.5, color="#bbbbbb", lw=0.6)
            prev = v
    fig.suptitle(f"周辺分布の変化：{old_label} → {new_label}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5)
    return _save(fig, path, dpi)


# ---------------------------------------------------------------------------
# まとめて出力
# ---------------------------------------------------------------------------
def plot_all(res, cfg, out_dir, formats=("png",), dpi=150, prefix=""):
    """network / marginals / updates を out_dir に書き、書いたパスのリストを返す。"""
    paths = []
    for fmt in formats:
        paths.append(plot_network(cfg, os.path.join(out_dir, f"{prefix}network.{fmt}"), dpi))
        paths.append(plot_marginals(res, cfg, os.path.join(out_dir, f"{prefix}marginals.{fmt}"), dpi))
        p = plot_updates(res, cfg, os.path.join(out_dir, f"{prefix}updates.{fmt}"), dpi)
        if p:
            paths.append(p)
    return paths
