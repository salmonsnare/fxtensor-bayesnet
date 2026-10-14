"""Regression tests for 0.11.2 (#26): a cross-check that was not run was reported as a max error of 0.0."""
import pytest

from fxtensor_bayesnet import run, to_english_keys
from fxtensor_bayesnet.report import render_markdown

L = ["s0", "s1", "s2"]
CPT = {"s0": [0.7, 0.2, 0.1], "s1": [0.2, 0.6, 0.2], "s2": [0.1, 0.3, 0.6]}


def chain(n):
    V = {f"v{i}": {"labels": L} for i in range(n)}
    K = {f"v{i}": {"parents": [f"v{i-1}"], "cpt": CPT} for i in range(1, n)}
    return {"as_of": "2026-10-15", "variables": V, "priors": {"v0": {"p": [0.5, 0.3, 0.2]}}, "kernels": K}


def ancestral(obs):
    c = chain(14)                                        # 3**14 elements > MAX_JOINT -> ancestral
    c["outputs"] = ["v13"]
    c["observations"] = obs
    return c


JOINT_OBS = [{"name": "joint", "evidence": {"v13": "s0", "v6": "s2"}, "focus": ["v0"]}]


def test_ancestral_joint_evidence_is_not_reported_as_checked():
    chk = run(ancestral(JOINT_OBS))["検算"]
    assert chk["計算モード"] == "ancestral"
    assert chk["einsum(条件付け)との最大差"] is None
    assert chk["条件付けの検算を省略した観測"] == ["joint"]
    assert chk["einsum(全同時)との最大差"] is None          # no full joint is built in ancestral mode
    assert isinstance(chk["einsum(祖先集合・周辺)との最大差"], float)   # this one is really computed
    en = to_english_keys({"検算": chk})["checks"]
    assert en["max_err_conditioning_vs_einsum"] is None and en["conditioning_check_skipped"] == ["joint"]


def test_ancestral_single_observation_still_checked():
    chk = run(ancestral([{"var": "v13", "label": "s0", "focus": ["v0"]}]))["検算"]
    assert isinstance(chk["einsum(条件付け)との最大差"], float) and "条件付けの検算を省略した観測" not in chk


def test_ancestral_posterior_matches_independent_calculation():
    c = ancestral(JOINT_OBS)
    post = run(c)["ベイズ更新(基本)"]["joint"]["事後"]["v0"]
    import numpy as np
    T = np.array([CPT[k] for k in L])
    p0 = np.array([0.5, 0.3, 0.2])
    a = p0[:, None] * np.linalg.matrix_power(T, 6)[:, 2][:, None]   # P(v0, v6=s2)
    w = a[:, 0] * (np.linalg.matrix_power(T, 7)[2, 0])                # × P(v13=s0 | v6=s2)
    w /= w.sum()
    assert [post[k] for k in L] == pytest.approx(list(w), abs=1e-4)


def test_joint_mode_values_are_numbers():
    c = chain(6)
    c["observations"] = [{"evidence": {"v5": "s0", "v3": "s2"}, "focus": ["v0"]}, {"var": "v5", "label": "s1"}]
    chk = run(c)["検算"]
    assert chk["計算モード"] == "joint"
    for k in ("einsum(全同時)との最大差", "einsum(祖先集合・周辺)との最大差", "einsum(条件付け)との最大差",
              "bayesian_inversionとconditionalizationの最大差"):
        assert isinstance(chk[k], float) and chk[k] < 1e-9
    assert "条件付けの検算を省略した観測" not in chk


def test_no_observations_means_not_checked():
    chk = run(chain(4))["検算"]
    assert chk["einsum(条件付け)との最大差"] is None
    assert chk["bayesian_inversionとconditionalizationの最大差"] is None
    assert isinstance(chk["einsum(全同時)との最大差"], float)


@pytest.mark.parametrize("lang,words", [("ja", ["未検算", "検算を省略した観測（einsum）: joint"]),
                                        ("en", ["Not checked", "max_err_conditioning_vs_einsum",
                                                "Observations not cross-checked (einsum): joint"])])
def test_markdown_states_unchecked(lang, words):
    c = ancestral(JOINT_OBS)
    md = render_markdown(run(c), c, lang=lang)
    for w in words:
        assert w in md
    c = chain(6)
    c["observations"] = [{"var": "v5", "label": "s1"}]
    md = render_markdown(run(c), c, lang=lang)
    assert "未検算" not in md and "Not checked" not in md
