"""Regression tests for 0.11.4 (#137): observing a root variable crashed the inversion cross-check (free_roots)."""
import json

import numpy as np
from fxtensor_bayesnet import run
from fxtensor_bayesnet.cli import main
from fxtensor_bayesnet.inference import Engine
from fxtensor_bayesnet.network import Net

# Issue #137: observing the root w made np.einsum reject "...->ab...a" (duplicate output letter).
ROOT_OBS = {
    "as_of": "2026-11-10",
    "variables": {"w": {"labels": ["弱い", "強い"]}, "d": {"labels": ["平常", "急増"]},
                  "l": {"labels": ["なし", "あり"]}},
    "priors": {"w": {"p": [0.45, 0.55]}, "d": {"p": [0.6, 0.4]}},
    "kernels": {"l": {"parents": ["w", "d"],
                      "cpt": {"弱い|平常": [0.8, 0.2], "強い|平常": [0.9, 0.1],
                              "弱い|急増": [0.4, 0.6], "強い|急増": [0.6, 0.4]}}},
    "observations": [{"var": "w", "label": "強い"}],
}

# Three roots (the observed one in the middle, three labels) and a chain below them.
THREE = {
    "as_of": "2026-11-10",
    "variables": {"a": {"labels": ["0", "1"]}, "b": {"labels": ["x", "y", "z"]}, "c": {"labels": ["L", "H"]},
                  "m": {"labels": ["lo", "hi"]}, "o": {"labels": ["n", "p"]}},
    "priors": {"a": {"p": [0.3, 0.7]}, "b": {"p": [0.2, 0.5, 0.3]}, "c": {"p": [0.6, 0.4]}},
    "kernels": {
        "m": {"parents": ["a", "b"], "cpt": {"0|x": [0.9, 0.1], "1|x": [0.7, 0.3], "0|y": [0.5, 0.5],
                                             "1|y": [0.4, 0.6], "0|z": [0.2, 0.8], "1|z": [0.1, 0.9]}},
        "o": {"parents": ["m", "c"], "cpt": {"lo|L": [0.8, 0.2], "hi|L": [0.3, 0.7],
                                             "lo|H": [0.6, 0.4], "hi|H": [0.1, 0.9]}},
    },
    "observations": [{"var": "b", "label": "z"}, {"var": "a", "label": "0"}, {"var": "o", "label": "p"}],
}


def test_root_observation_runs_and_posteriors_are_right():
    res = run(ROOT_OBS)
    up = res["ベイズ更新(基本)"]["w=強い を観測"]
    assert up["観測前の確率"] == 0.55
    assert up["事後"]["d"] == {"平常": 0.6, "急増": 0.4}            # roots are independent a priori
    assert up["事後"]["l"] == {"なし": 0.78, "あり": 0.22}          # 0.6*0.1 + 0.4*0.4
    err = res["検算"]["bayesian_inversionとconditionalizationの最大差"]
    assert err is not None and err < 1e-12


def test_kernel_with_root_target_is_delta():
    net = Net(THREE)
    pri = {r: THREE["priors"][r]["p"] for r in net.roots}
    k = net.einsum_query(pri, ["b"], free_roots=True)
    assert k.shape == (2, 3, 2, 3)
    np.testing.assert_allclose(k, np.broadcast_to(np.eye(3)[None, :, None, :], k.shape))
    # mixed targets: root b and non-root m; the axis order follows targets
    k2 = net.contract(pri, ["m", "b"], free_roots=True)
    km = net.contract(pri, ["m"], free_roots=True)
    assert k2.shape == (2, 3, 2, 2, 3)
    for j in range(3):
        np.testing.assert_allclose(k2[:, j, :, :, j], km[:, j, :, :])
        for jj in range(3):
            if jj != j:
                assert not k2[:, j, :, :, jj].any()


def test_inversion_matches_conditioning_for_root_and_nonroot_observations():
    net = Net(THREE)
    pri = {r: THREE["priors"][r]["p"] for r in net.roots}
    eng = Engine(net, pri, 10 ** 6)
    for obs, lab in (("b", "z"), ("a", "0"), ("c", "H"), ("o", "p"), ("m", "hi")):
        inv = eng.inversion_roots_raw(obs, lab)
        _, cond = eng.condition_raw(obs, lab)
        for r, a in inv.items():
            if r == obs:
                np.testing.assert_allclose(a, np.eye(len(net.labels[r]))[net.labels[r].index(lab)])
            else:
                np.testing.assert_allclose(a, cond[r], atol=1e-12)
    res = run(THREE)
    assert res["検算"]["bayesian_inversionとconditionalizationの最大差"] < 1e-12


def test_root_observation_cli_exits_0(tmp_path, capsys):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(ROOT_OBS, ensure_ascii=False), encoding="utf-8")
    assert main(["--config", str(p), "--no-write", "--md"]) == 0
    assert "w=強い を観測" in capsys.readouterr().out


def test_nonroot_free_roots_kernel_unchanged():
    net = Net(THREE)
    pri = {r: THREE["priors"][r]["p"] for r in net.roots}
    k = net.einsum_query(pri, ["o"], free_roots=True)
    assert k.shape == (2, 3, 2, 2)
    np.testing.assert_allclose(k.sum(axis=-1), 1.0)
