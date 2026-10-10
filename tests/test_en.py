"""English example configs and --lang en output."""
import os

import pytest

from fxtensor_bayesnet import load_config, run
from fxtensor_bayesnet.cli import main
from fxtensor_bayesnet.report import render_markdown

from conftest import EXAMPLES

PAIRS = [(os.path.join(EXAMPLES, n, "config.json"), os.path.join(EXAMPLES, n, "config.en.json"))
         for n in ("weather", "maintenance")]


@pytest.mark.parametrize("ja,en", PAIRS, ids=["weather", "maintenance"])
def test_english_config_is_numerically_identical(ja, en):
    cja, cen = load_config(ja), load_config(en)
    assert list(cja["variables"]) == list(cen["variables"])
    rja, ren = run(cja), run(cen)
    for (sj, a), (se, b) in zip(rja["シナリオ"].items(), ren["シナリオ"].items()):
        for v in a["周辺"]:
            assert list(a["周辺"][v].values()) == list(b["周辺"][v].values())
    for a, b in zip(rja["ベイズ更新(基本)"].values(), ren["ベイズ更新(基本)"].values()):
        assert a["観測前の確率"] == b["観測前の確率"]
        for w in a["事後"]:
            assert list(a["事後"][w].values()) == list(b["事後"][w].values())


def test_markdown_english():
    cfg = load_config(PAIRS[0][1])
    md = render_markdown(run(cfg), cfg, lang="en")
    assert md.startswith("# Results (as of 2026-10-10, Next day)")
    assert "| Scenario | None | Light | Heavy |" in md
    assert "| Base |" in md and "## Bayesian updates (base)" in md
    assert "### Observed rain=Heavy (prior probability" in md
    assert "基本" not in md and "観測" not in md


def test_cli_lang_en(capsys):
    assert main(["--config", PAIRS[1][1], "--no-write", "--md", "--lang", "en"]) == 0
    out = capsys.readouterr().out
    assert "## failure (Failure)" in out and "Ranking by movement (TV)" in out


def test_plot_english(tmp_path):
    pytest.importorskip("matplotlib")
    from fxtensor_bayesnet.plot import plot_all
    cfg = load_config(PAIRS[0][1])
    paths = plot_all(run(cfg), cfg, str(tmp_path), formats=("png",), lang="en")
    assert sorted(os.path.basename(p) for p in paths) == ["marginals.png", "network.png", "updates.png"]


def test_wrap_label():
    pytest.importorskip("matplotlib")
    from fxtensor_bayesnet.plot import _wrap_label
    assert _wrap_label("Low", 1.0) == "Low"
    assert _wrap_label("Stationary front", 0.6) == "Stationary\nfront"
