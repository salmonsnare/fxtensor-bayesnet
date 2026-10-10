"""図の出力（Agg バックエンド）と、matplotlib がないときのエラー。"""
import copy
import json
import subprocess
import sys

import pytest

from fxtensor_bayesnet import run
from fxtensor_bayesnet.cli import main
from fxtensor_bayesnet.report import render_mermaid

from conftest import EXAMPLE_CONFIGS, load

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def is_png(path):
    with open(path, "rb") as f:
        return f.read(8) == PNG_MAGIC


# ---------------------------------------------------------------------------
# matplotlib がある場合
# ---------------------------------------------------------------------------
@pytest.fixture
def plot():
    pytest.importorskip("matplotlib")
    from fxtensor_bayesnet import plot as plot_mod
    plot_mod._mpl()
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"
    return plot_mod


def test_each_figure_png_and_svg(plot, example_cfg, tmp_path):
    res = run(example_cfg)
    for fmt in ("png", "svg"):
        paths = [
            plot.plot_network(example_cfg, tmp_path / f"network.{fmt}"),
            plot.plot_marginals(res, example_cfg, tmp_path / f"marginals.{fmt}"),
            plot.plot_updates(res, example_cfg, tmp_path / f"updates.{fmt}"),
        ]
        for p in paths:
            assert p is not None and (tmp_path / p.name).stat().st_size > 1000
            if fmt == "png":
                assert is_png(p)
            else:
                assert "<svg" in open(p, encoding="utf-8").read(2000)


def test_diff_figure(plot, tmp_path):
    cfg = load(EXAMPLE_CONFIGS[0])
    new = copy.deepcopy(cfg)
    new["priors"]["pressure"]["p"] = [0.2, 0.3, 0.5]
    deltas = plot.diff_deltas(run(cfg), run(new))
    assert set(deltas) == {"基本", "A:台風接近", "B:乾燥した晴天続き"}
    assert any(abs(x) > 0.1 for _, _, x in deltas["基本"])
    # A・B は pressure を上書きするシナリオなので不変
    assert all(abs(x) < 1e-9 for _, _, x in deltas["A:台風接近"])
    assert all(abs(x) < 1e-9 for _, _, x in deltas["B:乾燥した晴天続き"])
    p = plot.plot_diff(run(cfg), run(new), new, tmp_path / "diff.png")
    assert is_png(p)


def test_updates_none_without_observations(plot, tmp_path):
    cfg = load(EXAMPLE_CONFIGS[1])
    cfg["observations"] = []
    assert plot.plot_updates(run(cfg), cfg, tmp_path / "u.png") is None


def test_layered_layout_uses_dummies_for_long_edges(plot):
    cfg = load(EXAMPLE_CONFIGS[1])            # load → temp は 2 層をまたぐ
    layers, level, parents, chains = plot.layered_layout(cfg)
    assert level["load"] == 0 and level["wear"] == 1 and level["temp"] == 2 and level["failure"] == 3
    assert len(chains[("load", "temp")]) == 1 and len(chains[("wear", "failure")]) == 1
    assert chains[("wear", "temp")] == []
    assert plot.node_kinds(cfg) == {"load": "root", "age": "root", "service": "root", "wear": "middle",
                                    "vibration": "output", "temp": "output", "failure": "output"}


def test_font_fallback_warns(plot, monkeypatch):
    monkeypatch.setattr(plot, "FONT_CANDIDATES", ["__no_such_font__"])
    monkeypatch.delenv(plot.FONT_ENV, raising=False)
    monkeypatch.setattr(plot, "_font", False)
    import matplotlib
    with matplotlib.rc_context():
        with pytest.warns(RuntimeWarning, match="日本語フォント"):
            assert plot.setup_font() is None


def test_font_env_override(plot, monkeypatch):
    from matplotlib import font_manager
    name = sorted(f.name for f in font_manager.fontManager.ttflist)[0]
    monkeypatch.setenv(plot.FONT_ENV, name)
    monkeypatch.setattr(plot, "_font", False)
    import matplotlib
    with matplotlib.rc_context():
        assert plot.setup_font() == name


def test_cli_plot_writes_files(plot, tmp_config, tmp_path, capsys):
    out = tmp_path / "figs"
    assert main(["--config", str(tmp_config), "--no-write", "--plot", "--plot-format", "png", "svg",
                 "--out-dir", str(out)]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["marginals.png", "marginals.svg", "network.png", "network.svg", "updates.png", "updates.svg"]
    assert "図を保存" in capsys.readouterr().err
    assert not (out / "results.json").exists()      # --no-write は結果ファイルを書かない


def test_cli_plot_with_diff(plot, tmp_config, capsys):
    assert main(["--config", str(tmp_config), "--no-write", "--snapshot"]) == 0
    cfg = json.loads(tmp_config.read_text(encoding="utf-8"))
    cfg["as_of"] = "2026-10-11"
    cfg["priors"]["humidity"]["p"] = [0.1, 0.4, 0.5]
    tmp_config.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    assert main(["--config", str(tmp_config), "--diff", "--plot"]) == 0
    assert is_png(tmp_config.parent / "diff.png")


def test_check_ignores_plot(tmp_config, capsys):
    assert main(["--config", str(tmp_config), "--check", "--plot"]) == 0
    assert not (tmp_config.parent / "network.png").exists()


# ---------------------------------------------------------------------------
# matplotlib がない場合
# ---------------------------------------------------------------------------
def test_missing_matplotlib_raises_plot_error(monkeypatch):
    from fxtensor_bayesnet import plot as plot_mod
    monkeypatch.setattr(plot_mod, "_plt", None)
    for m in [m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setitem(sys.modules, "matplotlib", None)       # import matplotlib が ImportError になる
    with pytest.raises(plot_mod.PlotError, match=r"pip install 'fxtensor-bayesnet\[plot\]'"):
        plot_mod._mpl()


def test_missing_matplotlib_cli_exit_code(tmp_config):
    """別プロセスで matplotlib を読めなくして CLI を実行：終了コード 3 と案内文。"""
    code = ("import sys; sys.modules['matplotlib'] = None; "
            "from fxtensor_bayesnet.cli import main; sys.exit(main(sys.argv[1:]))")
    r = subprocess.run([sys.executable, "-c", code, "--config", str(tmp_config), "--plot", "--no-write"],
                       capture_output=True, text=True)
    assert r.returncode == 3
    assert "matplotlib が必要です" in r.stderr and "fxtensor-bayesnet[plot]" in r.stderr
    # 図を使わない通常の実行は matplotlib なしでも動く
    r = subprocess.run([sys.executable, "-c", code, "--config", str(tmp_config), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# mermaid（既存出力の維持＋色分け）
# ---------------------------------------------------------------------------
def test_mermaid_kept_and_colored():
    cfg = load(EXAMPLE_CONFIGS[1])
    mmd = render_mermaid(cfg)
    assert mmd.startswith("graph TD")
    assert "load --> wear" in mmd and "wear --> failure" in mmd
    assert "class load,age,service root" in mmd
    assert "class wear middle" in mmd
    assert "class vibration,temp,failure output" in mmd
