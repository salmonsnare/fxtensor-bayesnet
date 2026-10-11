"""CPT sensitivity analysis (#5)."""
import copy

import pytest

from fxtensor_bayesnet import ConfigError, Net, run, validate
from fxtensor_bayesnet.causal import perturb_row, sensitivity

from test_bayes import CFG, hand_joint

PRI = {"A": [0.3, 0.7], "B": [0.6, 0.4]}


def test_perturb_row_keeps_distribution():
    q = perturb_row([0.2, 0.3, 0.5], 1, 0.4)
    assert q.sum() == pytest.approx(1) and q[1] == 0.4
    assert q[0] / q[2] == pytest.approx(0.2 / 0.5)                   # proportional co-variation
    assert list(perturb_row([0.0, 1.0, 0.0], 1, 0.9)) == pytest.approx([0.05, 0.9, 0.05])


def test_sensitivity_matches_hand_finite_difference():
    base, n, rows = sensitivity(Net(CFG), PRI, "C", "c1", delta=0.1, top=100)
    j = hand_joint()
    assert base == pytest.approx(j[:, :, 1, :].sum())
    # parameters: priors A, B (binary: 1 each) + 4 binary rows of C = 6; D is not an ancestor of C
    assert n == 6 and len(rows) == 6
    r = next(r for r in rows if r["パラメータ"] == "priors.A → a1")

    def pc1(pa1):
        return (1 - pa1) * (0.6 * 0.1 + 0.4 * 0.5) + pa1 * (0.6 * 0.6 + 0.4 * 0.8)

    assert r["P(下げたとき)"] == pytest.approx(pc1(0.6)) and r["P(上げたとき)"] == pytest.approx(pc1(0.8))
    r = next(r for r in rows if r["パラメータ"] == "kernels.C[a1|b1] → c1")
    assert r["幅(pt)"] == pytest.approx(0.7 * 0.4 * 0.2 * 100)
    assert [x["幅(pt)"] for x in rows] == sorted((x["幅(pt)"] for x in rows), reverse=True)


def test_clipping_at_bounds():
    _, _, rows = sensitivity(Net(CFG), PRI, "C", "c1", delta=0.5, top=100)
    r = next(r for r in rows if r["パラメータ"] == "kernels.C[a1|b1] → c1")    # 0.8 + 0.5 -> 1.0
    assert r["上げた値"] == 1.0 and r["下げた値"] == pytest.approx(0.3)


def test_multi_state_rows_perturb_each_label():
    _, n, _ = sensitivity(Net(CFG), PRI, "D", "d2", top=100)
    assert n == 2 + 4 + 2 * 3       # priors(2) + C rows(4) + D rows: 2 rows x 3 labels


def test_run_section():
    cfg = copy.deepcopy(CFG)
    cfg["sensitivity"] = [{"target": "D", "label": "d2", "top": 3}, {"target": "C"}]
    res = run(cfg)
    s = res["感度分析"]
    assert list(s) == ["D=d2", "C=c1"]
    assert len(s["D=d2"]["上位"]) == 3 and s["C=c1"]["変化量delta"] == 0.05


@pytest.mark.parametrize("sens,msg", [
    ({"target": "Q"}, "target は定義済み"),
    ({"target": "C", "label": "x"}, "ラベル x がない"),
    ({"target": "C", "delta": 1.5}, "delta"),
    ({"target": "C", "top": 0}, "top"),
])
def test_validation(sens, msg):
    cfg = copy.deepcopy(CFG)
    cfg["sensitivity"] = sens
    with pytest.raises(ConfigError, match=msg):
        validate(cfg)
