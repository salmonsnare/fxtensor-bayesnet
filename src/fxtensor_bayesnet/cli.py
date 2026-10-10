"""fxtensor-bayesnet — 設定ファイル駆動の離散ベイジアンネットワーク推論（fxtensor-salmon 使用）

使い方:
  fxtensor-bayesnet [--config PATH] [options]
    --config PATH     設定ファイル（既定: カレントディレクトリの config.json）
    --out-dir DIR     results.json / results.md / model.mmd の出力先（既定: 設定ファイルと同じフォルダ）
    --history-dir DIR スナップショットの保存先（既定: 設定ファイルと同じフォルダの history/）
    --no-write        results.json / results.md / model.mmd を書かず、標準出力だけにする
    --md              標準出力を Markdown にする（既定は JSON）
    --check           設定の検証だけ行う（エラーなら終了コード 2）
    --snapshot        設定と結果を <history-dir>/<as_of>/ に保存（既存なら --force が必要）
    --force           --snapshot の上書きを許可
    --diff [PATH]     設定（と結果）の前回スナップショットとの差分を表示
                      PATH 省略時は <history-dir> の as_of より前で最新のもの
    --format-config   設定ファイルを読みやすい形（核の1行=1行）に整形して書き戻す
    --plot            図を <out-dir> に出力（network / marginals / updates。--diff と併用で diff）
                      matplotlib が必要: pip install 'fxtensor-bayesnet[plot]'
    --plot-format F.. 図の形式 png / svg（複数可。既定: png）
    --plot-dpi N      PNG の解像度（既定: 150）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .config import ConfigError, config_hash, dump_config, load_config
from .history import diff_configs, diff_results, previous_snapshot, snapshot
from .inference import run
from .report import render_markdown, render_mermaid


def build_parser():
    ap = argparse.ArgumentParser(prog="fxtensor-bayesnet", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--history-dir", default=None)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--diff", nargs="?", const="__prev__")
    ap.add_argument("--format-config", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-format", nargs="+", choices=["png", "svg"], default=["png"])
    ap.add_argument("--plot-dpi", type=int, default=150)
    return ap


def _plot_module():
    """図のモジュールを読み込み、matplotlib の有無を先に確かめる。"""
    from . import plot
    plot._mpl()
    return plot


def main(argv=None):
    a = build_parser().parse_args(argv)
    base = os.path.dirname(os.path.abspath(a.config))
    out_dir = a.out_dir or base
    history_dir = a.history_dir or os.path.join(base, "history")

    try:
        cfg = load_config(a.config)
    except FileNotFoundError:
        print(f"設定ファイルがありません: {a.config}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"JSON の構文エラー: {a.config}: {e}", file=sys.stderr)
        return 2
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2

    plot = None
    if a.plot and not a.check:
        try:
            plot = _plot_module()
        except Exception as e:  # PlotError
            print(str(e), file=sys.stderr)
            return 3

    if a.format_config:
        with open(a.config, "w", encoding="utf-8") as f:
            f.write(dump_config(cfg))
        print(f"整形しました: {a.config}", file=sys.stderr)
    if a.check:
        print(f"設定OK: {a.config}（変数 {len(cfg['variables'])}、核 {len(cfg['kernels'])}、ハッシュ {config_hash(cfg)}）")
        return 0
    if a.diff:
        path = previous_snapshot(history_dir, cfg["as_of"]) if a.diff == "__prev__" else a.diff
        if not path:
            print(f"比較対象のスナップショットがありません（{history_dir}/<日付>/config.json）")
            return 0
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        print(f"# 設定の差分: {path} → {a.config}")
        print("\n".join(diff_configs(old, cfg)))
        rpath = os.path.join(os.path.dirname(path), "results.json")
        if os.path.exists(rpath):
            with open(rpath, encoding="utf-8") as f:
                old_res = json.load(f)
            new_res = run(cfg)
            print(f"\n# 結果の差分（0.1pt 以上）: {rpath} → 現在の設定で再計算")
            print("\n".join(diff_results(old_res, new_res)))
            if plot:
                for fmt in a.plot_format:
                    out = plot.plot_diff(old_res, new_res, cfg, os.path.join(out_dir, f"diff.{fmt}"), a.plot_dpi,
                                         old_label=old_res.get("基準日", "前回"), new_label=cfg["as_of"])
                    if out:
                        print(f"図を保存: {out}", file=sys.stderr)
        elif plot:
            print("前回の results.json がないため変化量の図は描けません", file=sys.stderr)
        return 0

    res = run(cfg)
    md = render_markdown(res, cfg)
    if not a.no_write:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
            f.write(md)
        with open(os.path.join(out_dir, "model.mmd"), "w", encoding="utf-8") as f:
            f.write(render_mermaid(cfg))
    if plot:
        for out in plot.plot_all(res, cfg, out_dir, a.plot_format, a.plot_dpi):
            print(f"図を保存: {out}", file=sys.stderr)
    if a.snapshot:
        try:
            d = snapshot(cfg, res, history_dir, a.force)
        except FileExistsError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(f"スナップショット保存: {d}", file=sys.stderr)
    print(md if a.md else json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
