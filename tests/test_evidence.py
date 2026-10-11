"""Joint conditioning on several observations and soft (likelihood) evidence (#4)."""
import copy

import numpy as np
import pytest

from fxtensor_bayesnet import ConfigError, Engine, Net, run, validate

from conftest import assert_close
from test_bayes import CFG, hand_joint

PRI = {"A": [0.3, 0.7], "B": [0.6, 0.4]}


def test_multiple_hard_evidence_matches_hand():
    j = hand_joint()
    ev, post = Engine(Net(CFG), PRI).posterior_raw({"C": "c1", "D": "d2"})
    sl = j[:, :, 1, 2]
    assert ev == pytest.approx(sl.sum())
    np.testing.assert_allclose(post["A"], sl.sum(axis=1) / sl.sum())
    np.testing.assert_allclose(post["B"], sl.sum(axis=0) / sl.sum())
    assert set(post) == {"A", "B"}


def test_soft_evidence_matches_hand():
    j = hand_joint()
    lik = np.array([0.2, 0.9])
    ev, post = Engine(Net(CFG), PRI).posterior_raw({"C": {"likelihood": lik.tolist()}, "D": "d0"})
    w = j[..., 0] * lik[None, None, :]
    assert ev == pytest.approx(w.sum())
    np.testing.assert_allclose(post["A"], w.sum(axis=(1, 2)) / w.sum())
    np.testing.assert_allclose(post["C"], w.sum(axis=(0, 1)) / w.sum())


def test_one_hot_likelihood_equals_hard_evidence():
    eng = Engine(Net(CFG), PRI)
    _, hard = eng.condition_raw("C", "c1")
    _, soft = eng.posterior_raw({"C": {"likelihood": [0, 1]}})
    for v in hard:
        np.testing.assert_allclose(hard[v], soft[v], atol=1e-12)


def test_run_with_evidence_entries():
    cfg = copy.deepcopy(CFG)
    cfg["observations"] = [
        {"var": "C", "label": "c1", "focus": ["A", "B"]},
        {"evidence": {"C": "c1"}, "focus": ["A", "B"]},
        {"evidence": {"C": "c1", "D": "d2"}, "focus": ["A", "B"]},
        {"name": "noisy sensor", "evidence": {"D": {"likelihood": [0.1, 0.3, 0.9]}}, "focus": ["A", "D"]},
    ]
    res = run(cfg)
    u = res["ベイズ更新(基本)"]
    assert list(u) == ["C=c1 を観測", "C=c1, D=d2 を観測", "noisy sensor"]   # same key for both single forms
    assert u["C=c1, D=d2 を観測"]["証拠"] == {"C": "c1", "D": "d2"}
    assert u["noisy sensor"]["証拠"] == {"D": {"尤度": [0.1, 0.3, 0.9]}}
    assert "D" in u["noisy sensor"]["注目変数の動き"]           # soft-evidence variables keep a posterior
    assert res["検算"]["einsum(条件付け)との最大差"] < 1e-12
    j = hand_joint()
    sl = j[:, :, 1, 2]
    assert u["C=c1, D=d2 を観測"]["観測前の確率"] == pytest.approx(sl.sum(), abs=1e-4)


def test_ancestral_mode_agrees():
    cfg = copy.deepcopy(CFG)
    cfg["observations"] = [{"evidence": {"C": "c1", "D": {"likelihood": [0.1, 0.3, 0.9]}}, "focus": ["A", "B"]}]
    assert_close(run(cfg)["ベイズ更新(基本)"], run(cfg, max_joint=0)["ベイズ更新(基本)"])


def test_zero_probability_evidence():
    cfg = copy.deepcopy(CFG)
    cfg["kernels"]["D"]["cpt"] = {"c0": [1, 0, 0], "c1": [0, 1, 0]}
    with pytest.raises(ValueError, match="確率が 0"):
        Engine(Net(cfg), PRI).posterior_raw({"D": "d2", "C": "c0"})


@pytest.mark.parametrize("ob,msg", [
    ({"evidence": {}}, "空でないオブジェクト"),
    ({"evidence": {"Z": "z"}}, "Z が未定義"),
    ({"evidence": {"C": "c9"}}, "ラベル c9 がない"),
    ({"evidence": {"C": {"likelihood": [0, 0]}}}, "少なくとも1つは正"),
    ({"evidence": {"C": {"likelihood": [0.5]}}}, "長さ 2"),
    ({"evidence": {"C": {"likelihood": [-1, 1]}}}, "非負"),
    ({"evidence": {"C": [0.5, 0.5]}}, "likelihood"),
    ({"evidence": {"C": "c1"}, "var": "C", "label": "c1"}, "同時に書けない"),
])
def test_validation(ob, msg):
    cfg = copy.deepcopy(CFG)
    cfg["observations"] = [ob]
    with pytest.raises(ConfigError, match=msg):
        validate(cfg)
