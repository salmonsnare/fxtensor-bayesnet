"""do-interventions on non-root nodes (graph surgery), ACE (#3), observed-vs-intervened figure (#7)."""
import copy

import numpy as np
import pytest

from fxtensor_bayesnet import ConfigError, do_config, load_config, run, validate
from fxtensor_bayesnet.cli import main

from conftest import ECOLOGY_CONFIG, assert_close
from test_bayes import CFG, hand_joint


def with_(cfg, **kw):
    c = copy.deepcopy(cfg)
    c.update(kw)
    return c


def hand_do_C(c_label_index):
    """P(D | do(C=c)) by truncated factorisation of the hand model."""
    pd = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    return pd[c_label_index]


def test_intervention_on_non_root_matches_truncated_factorisation():
    res = run(with_(CFG, interventions={"force c1": {"C": "c1"}, "mix": {"C": [0.25, 0.75]}}))
    iv = res["介入"]
    assert iv["force c1"]["介入内容"] == {"C": {"c0": 0.0, "c1": 1.0}}
    assert list(iv["force c1"]["周辺"]["D"].values()) == pytest.approx(hand_do_C(1), abs=1e-4)
    assert list(iv["mix"]["周辺"]["D"].values()) == pytest.approx(0.25 * hand_do_C(0) + 0.75 * hand_do_C(1), abs=1e-4)
    assert res["検算"]["介入:切断因子分解との最大差"] < 1e-12


def test_do_differs_from_conditioning_and_does_not_touch_parents():
    """Intervening on C leaves its parents A, B at their priors; conditioning on C moves them."""
    cfg = with_(CFG, interventions={"c1": {"C": "c1"}}, outputs=["A", "B", "C", "D"])
    res = run(cfg)
    assert res["介入"]["c1"]["周辺"]["A"] == {"a0": 0.3, "a1": 0.7}
    assert res["ベイズ更新(基本)"]["C=c1 を観測"]["事後"]["A"] != {"a0": 0.3, "a1": 0.7}


def test_intervention_on_root_equals_scenario():
    cfg = with_(CFG, scenarios={"A=a1": {"A": [0, 1]}}, interventions={"A=a1": {"A": "a1"}})
    res = run(cfg)
    assert res["介入"]["A=a1"]["周辺"] == res["シナリオ"]["A=a1"]["周辺"]


def test_do_config_is_pure_and_surgical():
    before = copy.deepcopy(CFG)
    d = do_config(CFG, {"C": "c0"})
    assert CFG == before
    assert "C" not in d["kernels"] and d["priors"]["C"]["p"] == [1.0, 0.0]
    validate(d)


def test_effects_ace_and_observed_difference():
    j = hand_joint()
    cfg = with_(CFG, effects=[{"treatment": "C", "outcome": "D", "outcome_label": "d2"},
                              {"treatment": "A", "outcome": "C"}])
    res = run(cfg)
    e = res["因果効果"]["C → D=d2"]
    assert (e["処置群"], e["対照群"]) == ("c1", "c0")
    assert e["ACE"] == pytest.approx(0.6 - 0.1, abs=1e-4)          # no confounding of C -> D
    p_obs = j[..., 2].sum(axis=(0, 1)) / j.sum(axis=(0, 1, 3))
    assert e["観測上の差"] == pytest.approx(p_obs[1] - p_obs[0], abs=1e-4)
    # A is a root with no confounding of A -> C: observational difference == ACE
    a = res["因果効果"]["A → C=c1"]
    assert a["ACE"] == pytest.approx(a["観測上の差"], abs=1e-4)
    assert a["交絡による差(観測上の差−ACE)"] == pytest.approx(0, abs=1e-4)


def test_ecology_example_reproduces_config_replacement_demo():
    """The values computed earlier by hand-made graph surgery (config replacement) for this model."""
    res = run(load_config(ECOLOGY_CONFIG))
    want = {  # treatment: (P(obs) Low, P(obs) High, P(do) Low, P(do) High)
        "parasitoid": (0.5968, 0.6454, 0.6055, 0.6342),
        "fly": (0.6795, 0.5492, 0.6786, 0.5504),
        "density": (0.6218, 0.6149, 0.6303, 0.6053),
    }
    for x, (ol, oh, dl, dh) in want.items():
        e = res["因果効果"][f"{x} → fledge=High"]
        assert e["P(結果|処置)(観測)"] == {"Low": ol, "High": oh}
        assert e["P(結果|do(処置))"] == {"Low": dl, "High": dh}
    assert round(res["因果効果"]["parasitoid → fledge=High"]["ACE"] * 100, 1) == 2.9
    assert round(res["因果効果"]["density → fledge=High"]["観測上の差"] * 100, 1) == -0.7
    assert round(res["因果効果"]["density → fledge=High"]["ACE"] * 100, 1) == -2.5


def test_interventions_in_ancestral_mode():
    cfg = load_config(ECOLOGY_CONFIG)
    a, b = run(cfg), run(cfg, max_joint=0)
    assert_close(a["介入"], b["介入"])
    assert_close(a["因果効果"], b["因果効果"])


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c.update(interventions={"x": {"Z": "a0"}}), "interventions.x: Z が未定義"),
    (lambda c: c.update(interventions={"x": {"C": "nope"}}), "ラベル nope がない"),
    (lambda c: c.update(interventions={"x": {"C": [0.5, 0.6]}}), "和が1でない"),
    (lambda c: c.update(interventions={"x": {}}), "空でないオブジェクト"),
    (lambda c: c.update(effects=[{"treatment": "C", "outcome": "C"}]), "同じ変数"),
    (lambda c: c.update(effects=[{"treatment": "C", "outcome": "Q"}]), "定義済みの変数"),
    (lambda c: c.update(effects=[{"treatment": "C", "outcome": "D", "treated": "c0"}]), "同じラベル"),
    (lambda c: c.update(effects=[{"treatment": "C", "outcome": "D", "outcome_label": "zz"}]), "ラベル zz がない"),
    (lambda c: c.update(scenarios={"s": {"C": [0.5, 0.5]}}), "interventions を使う"),
])
def test_validation(mutate, msg):
    c = copy.deepcopy(CFG)
    mutate(c)
    with pytest.raises(ConfigError, match=msg):
        validate(c)


def test_markdown_and_plot(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    assert main(["--config", ECOLOGY_CONFIG, "--out-dir", str(tmp_path), "--md", "--plot", "--lang", "en"]) == 0
    out = capsys.readouterr().out
    assert "## Interventions (do: graph surgery)" in out and "## Causal effects" in out
    assert "average causal effect ACE +2.9pt" in out
    assert (tmp_path / "effects.png").stat().st_size > 1000
    assert (tmp_path / "sensitivity.png").stat().st_size > 1000


def test_no_new_sections_without_new_keys(example_cfg):
    res = run(example_cfg)
    assert not {"介入", "因果効果", "感度分析"} & set(res)
