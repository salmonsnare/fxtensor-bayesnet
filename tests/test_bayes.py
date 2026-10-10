"""ベイズ更新（条件付け）と要因の逆算（bayesian_inversion）を手計算と照合。"""
import numpy as np
import pytest

from fxtensor_bayesnet import Engine, Net, run

# 原因 A, B（根）→ 結果 C の小さなネットワーク
CFG = {
    "as_of": "2026-01-01",
    "variables": {
        "A": {"name": "原因A", "labels": ["a0", "a1"]},
        "B": {"name": "原因B", "labels": ["b0", "b1"]},
        "C": {"name": "結果", "labels": ["c0", "c1"]},
        "D": {"name": "結果の結果", "labels": ["d0", "d1", "d2"]},
    },
    "priors": {"A": {"p": [0.3, 0.7]}, "B": {"p": [0.6, 0.4]}},
    "kernels": {
        "C": {"parents": ["A", "B"], "cpt": {
            "a0|b0": [0.9, 0.1], "a0|b1": [0.5, 0.5], "a1|b0": [0.4, 0.6], "a1|b1": [0.2, 0.8]}},
        "D": {"parents": ["C"], "cpt": {"c0": [0.7, 0.2, 0.1], "c1": [0.1, 0.3, 0.6]}},
    },
    "observations": [{"var": "C", "label": "c1", "focus": ["A", "B"]}],
    "outputs": ["C", "D"],
}


def hand_joint():
    pa, pb = np.array([0.3, 0.7]), np.array([0.6, 0.4])
    pc = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.4, 0.6], [0.2, 0.8]]])
    pd = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    return np.einsum("a,b,abc,cd->abcd", pa, pb, pc, pd)


def test_marginal_by_hand():
    j = hand_joint()
    eng = Engine(Net(CFG), {"A": [0.3, 0.7], "B": [0.6, 0.4]})
    np.testing.assert_allclose(eng.marginal("C"), j.sum(axis=(0, 1, 3)))
    np.testing.assert_allclose(eng.marginal("D"), j.sum(axis=(0, 1, 2)))


def test_condition_matches_bayes_rule():
    j = hand_joint()
    eng = Engine(Net(CFG), {"A": [0.3, 0.7], "B": [0.6, 0.4]})
    ev, post = eng.condition("C", "c1")
    pc1 = j[:, :, 1, :].sum()
    assert ev == pytest.approx(pc1)
    want_a = j[:, :, 1, :].sum(axis=(1, 2)) / pc1
    want_b = j[:, :, 1, :].sum(axis=(0, 2)) / pc1
    want_d = j[:, :, 1, :].sum(axis=(0, 1)) / pc1
    assert list(post["A"].values()) == pytest.approx(want_a, abs=1e-4)
    assert list(post["B"].values()) == pytest.approx(want_b, abs=1e-4)
    assert list(post["D"].values()) == pytest.approx(want_d, abs=1e-4)


def test_condition_on_downstream_observation():
    """孫の観測 D=d2 から根を逆算。"""
    j = hand_joint()
    eng = Engine(Net(CFG), {"A": [0.3, 0.7], "B": [0.6, 0.4]})
    ev, post = eng.condition("D", "d2")
    sl = j[..., 2]
    assert ev == pytest.approx(sl.sum())
    assert list(post["A"].values()) == pytest.approx(sl.sum(axis=(1, 2)) / sl.sum(), abs=1e-4)
    assert list(post["C"].values()) == pytest.approx(sl.sum(axis=(0, 1)) / sl.sum(), abs=1e-4)


def test_inversion_roots_matches_conditionalization():
    eng = Engine(Net(CFG), {"A": [0.3, 0.7], "B": [0.6, 0.4]})
    _, post = eng.condition("C", "c1")
    inv = eng.inversion_roots("C", "c1")
    for r in ("A", "B"):
        assert list(inv[r].values()) == pytest.approx(list(post[r].values()), abs=1e-4)


def test_run_reports_movement_and_ranking():
    res = run(CFG)
    upd = res["ベイズ更新(基本)"]["C=c1 を観測"]
    j = hand_joint()
    assert upd["観測前の確率"] == pytest.approx(j[:, :, 1, :].sum(), abs=1e-4)
    mv = upd["注目変数の動き"]
    post_a = j[:, :, 1, :].sum(axis=(1, 2)) / j[:, :, 1, :].sum()
    tv_a = 0.5 * np.abs(post_a - np.array([0.3, 0.7])).sum()
    assert mv["A"]["全変動距離TV"] == pytest.approx(tv_a, abs=1e-4)
    assert set(upd["動きの大きさ順(TV)"]) == {"A", "B"}
    assert mv[upd["動きの大きさ順(TV)"][0]]["全変動距離TV"] >= mv[upd["動きの大きさ順(TV)"][1]]["全変動距離TV"]


def test_example_observations_shift_causes_upward(example_cfg):
    """例題の観測（悪い結果）で、注目変数の事後が事前と異なり、確率の和は 1。"""
    res = run(example_cfg)
    for r in res["ベイズ更新(基本)"].values():
        for w, d in r["事後"].items():
            assert sum(d.values()) == pytest.approx(1, abs=1e-3)
        assert any(m["全変動距離TV"] > 0 for m in r["注目変数の動き"].values())
