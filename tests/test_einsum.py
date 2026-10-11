"""ライブラリ（FXTensor）での計算と numpy einsum の直接計算の突き合わせ。"""
import numpy as np

from fxtensor_bayesnet import Engine, Net, run
from fxtensor_bayesnet.network import graph_extend, self_test_graph_extend

from conftest import assert_close


def base_priors(cfg):
    return {v: cfg["priors"][v]["p"] for v in cfg["priors"]}


def test_graph_extend_matches_library_composition():
    self_test_graph_extend()


def test_joint_matches_brute_force_einsum(example_cfg):
    net = Net(example_cfg)
    pri = base_priors(example_cfg)
    joint, names = net.build_joint(pri)
    perm = [names.index(v) for v in net.order]
    np.testing.assert_allclose(np.transpose(joint.data, perm), net.brute_force_joint(pri), atol=1e-12)
    assert abs(joint.data.sum() - 1) < 1e-12


def test_marginals_match_einsum_for_all_scenarios(example_cfg):
    net = Net(example_cfg)
    pri0 = base_priors(example_cfg)
    for ov in [{}] + list(example_cfg.get("scenarios", {}).values()):
        pri = {**pri0, **ov}
        eng = Engine(net, pri)
        assert eng.mode == "joint"
        full = net.brute_force_joint(pri)
        for i, v in enumerate(net.order):
            axes = tuple(j for j in range(len(net.order)) if j != i)
            np.testing.assert_allclose(eng.marginal(v), full.sum(axis=axes), atol=1e-12)
            np.testing.assert_allclose(eng.marginal(v), net.einsum_query(pri, [v]), atol=1e-12)


def test_ancestral_mode_gives_same_results(example_cfg):
    """同時分布が大きいとき用の「祖先集合 einsum」経路でも結果が同じ。"""
    a = run(example_cfg)
    b = run(example_cfg, max_joint=1)
    assert a["検算"]["計算モード"] == "joint"
    assert b["検算"]["計算モード"] == "ancestral"
    assert_close(a["シナリオ"], b["シナリオ"])
    for k in a["ベイズ更新(基本)"]:
        assert_close(a["ベイズ更新(基本)"][k]["事後"], b["ベイズ更新(基本)"][k]["事後"])


def test_run_self_checks_are_small(example_cfg):
    chk = run(example_cfg)["検算"]
    assert chk["einsum(全同時)との最大差"] < 1e-12
    assert chk["einsum(祖先集合・周辺)との最大差"] < 1e-12
    assert chk["einsum(条件付け)との最大差"] < 1e-12
    assert chk["bayesian_inversionとconditionalizationの最大差"] < 1e-12


def test_graph_extend_shape():
    from fxtensor_salmon import FXTensor
    st = FXTensor([[], [["x0", "x1"]]], data=np.array([0.4, 0.6]))
    k = FXTensor([[["x0", "x1"]], [["y0", "y1", "y2"]]], data=np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]))
    out = graph_extend(st, k, [0])
    assert out.data.shape == (2, 3)
    np.testing.assert_allclose(out.data, np.array([0.4, 0.6])[:, None] * k.data)
