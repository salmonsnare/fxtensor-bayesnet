"""Regression tests for the bugs fixed in 0.10.1 (#17 soft-evidence label, #18 zero-probability observation,
#19 long observation names in updates.png)."""
import copy
import json

import pytest

from fxtensor_bayesnet import run, to_english_keys
from fxtensor_bayesnet.cli import main
from fxtensor_bayesnet.report import render_markdown

L2 = ["Low", "High"]


# --- #17: soft evidence reports an expected likelihood, not a probability ------------------
SOFT = {
    "as_of": "2026-10-12",
    "variables": {"rain": {"name": "Rain", "labels": ["No", "Yes"]}, "wet": {"name": "Ground wet", "labels": ["Dry", "Wet"]}},
    "priors": {"rain": {"p": [0.4, 0.6]}},
    "kernels": {"wet": {"parents": ["rain"], "cpt": {"No": [0.8, 0.2], "Yes": [0.1, 0.9]}}},
    "outputs": ["wet"],
    "observations": [{"name": "soft", "evidence": {"rain": {"likelihood": [2, 9]}}},
                     {"name": "hard", "evidence": {"wet": "Wet"}},
                     {"var": "wet", "label": "Wet"}],
}


def test_soft_evidence_is_reported_as_expected_likelihood():
    res = run(SOFT)
    soft = res["ベイズ更新(基本)"]["soft"]
    assert "観測前の確率" not in soft
    assert soft["尤度の期待値"] == pytest.approx(0.4 * 2 + 0.6 * 9)          # 6.2 > 1
    assert soft["事後"]["rain"] == {"No": pytest.approx(0.8 / 6.2, abs=1e-4), "Yes": pytest.approx(5.4 / 6.2, abs=1e-4)}
    # hard evidence keeps the probability (both forms)
    for k in ("hard", "wet=Wet を観測"):
        assert res["ベイズ更新(基本)"][k]["観測前の確率"] == pytest.approx(0.4 * 0.2 + 0.6 * 0.9)
        assert "尤度の期待値" not in res["ベイズ更新(基本)"][k]
    en = to_english_keys(res)["updates_base"]
    assert en["soft"]["expected_likelihood"] == soft["尤度の期待値"] and "evidence_probability" not in en["soft"]
    assert "evidence_probability" in en["hard"]


@pytest.mark.parametrize("lang, soft_head, hard_head", [
    ("ja", "### soft を観測（尤度の期待値 6.2）", "### hard を観測（観測前の確率 62.0%）"),
    ("en", "### Observed soft (expected likelihood 6.2)", "### Observed hard (prior probability 62.0%)"),
])
def test_soft_evidence_markdown_does_not_show_a_percentage(lang, soft_head, hard_head):
    md = render_markdown(run(SOFT), SOFT, lang)
    assert soft_head in md and hard_head in md
    assert "620.0%" not in md


# --- #18: one zero-probability observation does not abort the run ---------------------------
ZERO = {
    "as_of": "2026-10-12",
    "variables": {"nest": {"name": "Nest density", "labels": L2}, "fly": {"name": "Parasitic fly", "labels": L2},
                  "fledge": {"name": "Fledging", "labels": L2}},
    "priors": {"nest": {"p": [0.5, 0.5]}},
    "kernels": {"fly": {"parents": ["nest"], "cpt": {"Low": [1.0, 0.0], "High": [1.0, 0.0]}},   # P(fly=High) = 0
                "fledge": {"parents": ["fly"], "cpt": {"Low": [0.3, 0.7], "High": [0.6, 0.4]}}},
    "outputs": ["fledge"],
    "observations": [{"name": "ok", "evidence": {"fly": "Low"}, "focus": ["fledge"]},
                     {"name": "impossible", "evidence": {"fly": "High", "fledge": "Low"}},
                     {"var": "fly", "label": "High"},
                     {"name": "impossible soft", "evidence": {"fly": {"likelihood": [0, 1]}}}],
}


def test_zero_probability_observation_is_recorded_and_others_still_computed():
    with pytest.warns(UserWarning, match="確率 0"):
        res = run(ZERO)
    u = res["ベイズ更新(基本)"]
    assert list(u) == ["ok", "impossible", "fly=High を観測", "impossible soft"]
    assert u["ok"]["事後"]["fledge"] == {"Low": 0.3, "High": 0.7} and "不可能な観測" not in u["ok"]
    for k in ("impossible", "fly=High を観測"):
        assert u[k]["不可能な観測"] is True and u[k]["観測前の確率"] == 0.0 and u[k]["事後"] == {}
        assert u[k]["注目変数の事前"] == {"nest": {"Low": 0.5, "High": 0.5}}
    assert u["impossible soft"]["不可能な観測"] is True and u["impossible soft"]["尤度の期待値"] == 0.0
    assert u["impossible"]["証拠"] == {"fly": "High", "fledge": "Low"}
    assert to_english_keys(res)["updates_base"]["impossible"]["impossible"] is True
    assert res["シナリオ"]["基本"]["周辺"]["fledge"] == {"Low": 0.3, "High": 0.7}
    for lang, msg in (("ja", "確率 0 の観測"), ("en", "probability 0")):
        assert msg in render_markdown(res, ZERO, lang)


def test_zero_probability_observation_cli_exits_0(tmp_path, capsys):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(ZERO), encoding="utf-8")
    with pytest.warns(UserWarning):
        assert main(["--config", str(p), "--no-write", "--md", "--lang", "en"]) == 0
    assert "### Observed ok" in capsys.readouterr().out


def test_zero_probability_observation_plot(tmp_path):
    pytest.importorskip("matplotlib")
    from fxtensor_bayesnet import plot
    with pytest.warns(UserWarning):
        res = run(ZERO)
    assert plot.plot_updates(res, ZERO, tmp_path / "updates.png", lang="en").stat().st_size > 1000
    only_impossible = {**res, "ベイズ更新(基本)": {"impossible": res["ベイズ更新(基本)"]["impossible"]}}
    assert plot.plot_updates(only_impossible, ZERO, tmp_path / "none.png") is None


# --- #19: long observation names do not squash the panels of updates.png --------------------
def _axes_widths(monkeypatch, plot, cfg, lang):
    widths = {}

    def fake_save(fig, path, dpi):
        fig.canvas.draw()
        widths["w"] = [ax.get_position().width * fig.get_figwidth() for ax in fig.axes if ax.axison]
        plot._mpl().close(fig)
        return path

    monkeypatch.setattr(plot, "_save", fake_save)
    plot.plot_updates(run(cfg), cfg, "unused.png", lang=lang)
    return widths["w"]


@pytest.mark.parametrize("lang, long_name", [
    ("en", "Visual fly count with imperfect detection (soft evidence), second survey round"),
    ("ja", "目視による寄生バエの個体数調査（検出漏れを含むソフト証拠）"),
])
def test_long_observation_name_keeps_panel_width(monkeypatch, lang, long_name):
    pytest.importorskip("matplotlib")
    from fxtensor_bayesnet import plot
    L = {"name": "n", "labels": L2}
    cfg = {"as_of": "2026-10-12", "variables": {"a": L, "c": {**L, "name": "Other"}, "b": {**L, "name": "Outcome"}},
           "priors": {"a": {"p": [0.4, 0.6]}, "c": {"p": [0.5, 0.5]}},
           "kernels": {"b": {"parents": ["a", "c"], "cpt": {"Low|Low": [0.7, 0.3], "Low|High": [0.6, 0.4],
                                                           "High|Low": [0.2, 0.8], "High|High": [0.1, 0.9]}}},
           "outputs": ["b"],
           "observations": [{"name": "S", "evidence": {"b": {"likelihood": [0.2, 0.9]}}, "focus": ["a", "c"]},
                            {"name": "T", "evidence": {"b": "High"}, "focus": ["a"]}]}
    short = _axes_widths(monkeypatch, plot, cfg, lang)
    long_cfg = copy.deepcopy(cfg)
    long_cfg["observations"][0]["name"] = long_name
    long_ = _axes_widths(monkeypatch, plot, long_cfg, lang)
    assert len(short) == len(long_) == 3
    assert min(long_) > 0.9 * min(short) and min(long_) > 2.0      # inches; was ~0.3 in 0.10.0
