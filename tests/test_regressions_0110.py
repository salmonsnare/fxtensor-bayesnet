"""Tests for 0.11.0: #21 undefined observed probability drawn as a 0 bar in effects.png, #22 stratified (given_do)
and conditional (given) ACE in effects."""
import copy
import math
import warnings

import pytest

from fxtensor_bayesnet import run, to_english_keys
from fxtensor_bayesnet.config import ConfigError, validate
from fxtensor_bayesnet.report import render_markdown

# --- #21: a treatment level with probability 0 -----------------------------------------------
ZERO = {
    "as_of": "2026-10-12",
    "variables": {"x": {"labels": ["a", "b", "c"]}, "y": {"labels": ["n", "p"]}},
    "priors": {"x": {"p": [0.5, 0.5, 0.0]}},
    "kernels": {"y": {"parents": ["x"], "cpt": {"a": [0.8, 0.2], "b": [0.5, 0.5], "c": [0.1, 0.9]}}},
    "effects": [{"treatment": "x", "outcome": "y", "outcome_label": "p", "treated": "c", "control": "a"}],
}


def _run_zero():
    with pytest.warns(UserWarning, match="未定義"):
        return run(ZERO)


def test_zero_level_results_and_warning():
    e = _run_zero()["因果効果"]["x → y=p"]
    assert e["P(結果|処置)(観測)"]["c"] is None and e["観測上の差"] is None
    assert e["ACE"] == pytest.approx(0.7)                       # do() side is defined
    assert e["観測確率0の水準"] == ["c"]
    assert to_english_keys({"因果効果": {"k": e}})["effects"]["k"]["zero_probability_levels"] == ["c"]


def test_no_zero_level_key_or_warning_when_all_levels_possible():
    cfg = copy.deepcopy(ZERO)
    cfg["priors"]["x"]["p"] = [0.4, 0.4, 0.2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e = run(cfg)["因果効果"]["x → y=p"]
    assert "観測確率0の水準" not in e and set(e) == {
        "処置", "結果", "結果のラベル", "処置群", "対照群", "P(結果|do(処置))", "P(結果|処置)(観測)", "ACE",
        "観測上の差", "交絡による差(観測上の差−ACE)"}   # unchanged output when the new keys are not used


@pytest.mark.parametrize("lang,word", [("ja", "未定義"), ("en", "undefined")])
def test_zero_level_markdown(lang, word):
    md = render_markdown(_run_zero(), ZERO, lang=lang)
    sec = md[md.index("x → y=p"):]
    assert "| c | — | 90.0% |" in sec and word in sec


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_zero_level_not_drawn_as_zero_bar(monkeypatch, lang):
    pytest.importorskip("matplotlib")
    from fxtensor_bayesnet import plot
    res = _run_zero()
    seen = {}

    def fake_save(fig, path, dpi):
        ax = fig.axes[0]
        seen["heights"] = [p.get_height() for p in ax.patches]
        seen["texts"] = [t.get_text() for t in ax.texts]
        plot._mpl().close(fig)
        return path

    monkeypatch.setattr(plot, "_save", fake_save)
    plot.plot_effects(res, ZERO, "unused.png", lang=lang)
    obs_heights = seen["heights"][:3]               # first series = observed, levels a, b, c
    assert obs_heights[:2] == pytest.approx([20, 50]) and math.isnan(obs_heights[2])
    # no "0" annotation: a, b, c observed = 20, 50, (undefined); do = 20, 50, 90
    assert sorted(x for x in seen["texts"] if x.isdigit()) == ["20", "20", "50", "50", "90"]
    assert any(("未定義" if lang == "ja" else "undefined") in x for x in seen["texts"])


# --- #22: stratified / conditional ACE ---------------------------------------------------------
GARDEN = {
    "as_of": "2026-10-12",
    "variables": {
        "weather": {"labels": ["Dry", "Wet"]},
        "water": {"labels": ["No", "Yes"]},
        "soil": {"labels": ["Poor", "Rich"]},
        "harvest": {"labels": ["Bad", "Good"]},
    },
    "priors": {"weather": {"p": [0.4, 0.6]}},
    "kernels": {
        "water": {"parents": ["weather"], "cpt": {"Dry": [0.3, 0.7], "Wet": [0.8, 0.2]}},
        "soil": {"parents": ["weather"], "cpt": {"Dry": [0.6, 0.4], "Wet": [0.3, 0.7]}},
        "harvest": {"parents": ["weather", "water", "soil"], "cpt": {
            "Dry|No|Poor": [0.9, 0.1], "Dry|No|Rich": [0.7, 0.3], "Dry|Yes|Poor": [0.5, 0.5],
            "Dry|Yes|Rich": [0.2, 0.8], "Wet|No|Poor": [0.6, 0.4], "Wet|No|Rich": [0.3, 0.7],
            "Wet|Yes|Poor": [0.5, 0.5], "Wet|Yes|Rich": [0.25, 0.75]}},
    },
    "outputs": ["harvest"],
    "effects": [{"treatment": "water", "outcome": "harvest", "outcome_label": "Good",
                 "given_do": {"weather": ["Dry", "Wet"]}, "given": {"soil": None}}],
}


def test_given_do_matches_four_interventions():
    cfg = copy.deepcopy(GARDEN)
    cfg["interventions"] = {f"{w}|{s}": {"water": w, "weather": s} for w in ("No", "Yes") for s in ("Dry", "Wet")}
    res = run(cfg)
    p = {k: r["周辺"]["harvest"]["Good"] for k, r in res["介入"].items()}
    st = res["因果効果"]["water → harvest=Good"]["層別ACE(given_do)"]
    assert st["条件"] == ["weather"] and list(st["層"]) == ["weather=Dry", "weather=Wet"]
    for s in ("Dry", "Wet"):
        layer = st["層"][f"weather={s}"]
        assert layer["条件の値"] == {"weather": s}
        assert layer["P(結果|do(処置),do(層))"] == {"No": pytest.approx(p[f"No|{s}"]), "Yes": pytest.approx(p[f"Yes|{s}"])}
        assert layer["ACE"] == pytest.approx(p[f"Yes|{s}"] - p[f"No|{s}"], abs=1e-4)
    # by hand: Dry: soil Poor 0.6 / Rich 0.4 → No 0.6*0.1+0.4*0.3=0.18, Yes 0.6*0.5+0.4*0.8=0.62
    assert st["層"]["weather=Dry"]["ACE"] == pytest.approx(0.44)
    assert st["層"]["weather=Wet"]["ACE"] == pytest.approx(0.675 - 0.61)
    assert st["層間のACEの差(最後−最初)"] == pytest.approx(0.065 - 0.44)


def test_given_matches_conditioning_in_mutilated_model():
    res = run(GARDEN)
    st = res["因果効果"]["water → harvest=Good"]["条件付きACE(given)"]
    assert list(st["層"]) == ["soil=Poor", "soil=Rich"]
    for s in ("Poor", "Rich"):
        hand = {}
        for w in ("No", "Yes"):
            # graph surgery by hand: water fixed, then observe soil
            cfg = copy.deepcopy(GARDEN)
            cfg.pop("effects")
            del cfg["kernels"]["water"]
            cfg["priors"]["water"] = {"p": [1.0, 0.0] if w == "No" else [0.0, 1.0]}
            cfg["observations"] = [{"name": "z", "evidence": {"soil": s}}]
            hand[w] = run(cfg)["ベイズ更新(基本)"]["z"]["事後"]["harvest"]["Good"]
        layer = st["層"][f"soil={s}"]
        assert layer["P(結果|do(処置),層)"] == {"No": pytest.approx(hand["No"]), "Yes": pytest.approx(hand["Yes"])}
        assert layer["ACE"] == pytest.approx(hand["Yes"] - hand["No"], abs=1e-4)
    # explicit arithmetic for soil=Poor, do(water=Yes): (0.4*0.6*0.5 + 0.6*0.3*0.5) / (0.4*0.6 + 0.6*0.3) = 0.5
    assert st["層"]["soil=Poor"]["P(結果|do(処置),層)"]["Yes"] == pytest.approx(0.5)
    assert st["層"]["soil=Poor"]["P(層|do(処置))"]["Yes"] == pytest.approx(0.42)


def test_given_undefined_stratum_is_null():
    cfg = copy.deepcopy(GARDEN)
    cfg["kernels"]["soil"]["cpt"] = {"Dry": [1.0, 0.0], "Wet": [1.0, 0.0]}   # soil=Rich never happens
    cfg["effects"] = [{"treatment": "water", "outcome": "harvest", "given": {"soil": ["Poor", "Rich"]}}]
    with pytest.warns(UserWarning, match="soil=Rich"):
        res = run(cfg)
    st = res["因果効果"]["water → harvest=Good"]["条件付きACE(given)"]
    assert st["層"]["soil=Rich"]["ACE"] is None and st["層間のACEの差(最後−最初)"] is None
    assert st["層"]["soil=Poor"]["ACE"] is not None
    assert "— " in render_markdown(res, cfg, lang="en")


def test_multiple_given_do_variables_make_combined_strata():
    cfg = copy.deepcopy(GARDEN)
    cfg["effects"] = [{"treatment": "water", "outcome": "harvest", "given_do": {"weather": None, "soil": "Rich"}}]
    st = run(cfg)["因果効果"]["water → harvest=Good"]["層別ACE(given_do)"]
    assert list(st["層"]) == ["weather=Dry, soil=Rich", "weather=Wet, soil=Rich"]
    assert st["層"]["weather=Dry, soil=Rich"]["ACE"] == pytest.approx(0.8 - 0.3)


def test_english_keys_and_markdown():
    res = run(GARDEN)
    en = to_english_keys(res)["effects"]["water → harvest=Good"]
    sd = en["stratified_ace_given_do"]
    assert sd["given"] == ["weather"] and sd["strata"]["weather=Dry"]["given_values"] == {"weather": "Dry"}
    assert set(sd["strata"]["weather=Dry"]) == {"given_values", "p_outcome_do_given_do", "ace"}
    assert "ace_difference_last_minus_first" in sd
    assert set(en["conditional_ace_given"]["strata"]["soil=Poor"]) == {
        "given_values", "p_stratum_do", "p_outcome_do_given", "ace"}
    for lang, a, b in (("ja", "層別 ACE", "条件付き ACE"), ("en", "Stratified ACE", "Conditional ACE")):
        md = render_markdown(res, GARDEN, lang=lang)
        assert a in md and b in md and "| weather=Dry | 18.0% | 62.0% | +44.0pt |" in md
        assert "-37.5pt" in md


def test_strata_plot_panels(monkeypatch):
    pytest.importorskip("matplotlib")
    from fxtensor_bayesnet import plot
    res = run(GARDEN)
    seen = {}

    def fake_save(fig, path, dpi):
        seen["n"] = sum(1 for ax in fig.axes if ax.axison)
        seen["bars"] = [round(p.get_height(), 1) for p in fig.axes[1].patches]
        plot._mpl().close(fig)
        return path

    monkeypatch.setattr(plot, "_save", fake_save)
    plot.plot_effects(res, GARDEN, "unused.png")
    assert seen["n"] == 3 and seen["bars"] == [44.0, 6.5]


@pytest.mark.parametrize("bad,msg", [
    ({"given_do": {"nope": None}}, "未定義"),
    ({"given_do": {"water": None}}, "別の変数"),
    ({"given": {"harvest": None}}, "別の変数"),
    ({"given_do": {"weather": ["Hot"]}}, "ラベル Hot"),
    ({"given_do": {"weather": None}, "given": {"weather": None}}, "両方"),
    ({"given": []}, "辞書"),
])
def test_given_validation(bad, msg):
    cfg = copy.deepcopy(GARDEN)
    cfg["effects"] = [{"treatment": "water", "outcome": "harvest", **bad}]
    with pytest.raises(ConfigError, match=msg):
        validate(cfg)


@pytest.mark.parametrize("readme,section,cfgname", [
    ("README.md", "### Stratified and conditional ACE", "config.en.json"),
    ("README-ja.md", "### 層別 ACE・条件付き ACE", "config.json")], ids=["en", "ja"])
def test_readme_given_snippet(readme, section, cfgname):
    import json
    import os
    import re

    from conftest import ROOT

    from fxtensor_bayesnet import load_config
    with open(os.path.join(ROOT, readme), encoding="utf-8") as f:
        text = f.read()
    text = text[text.index(section):]
    item = json.loads(re.search(r"```json\n(.*?)```", text, re.DOTALL).group(1))
    cfg = load_config(os.path.join(ROOT, "examples", "maintenance", cfgname))
    cfg["effects"] = [item]
    validate(cfg)
    res = run(cfg)
    md = render_markdown(res, cfg, lang="en" if readme == "README.md" else "ja")
    excerpt = re.search(r"```\n(#### .*?)```", text, re.DOTALL).group(1).strip()
    assert excerpt in md
