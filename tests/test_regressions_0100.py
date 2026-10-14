"""Regression tests for the bugs fixed in 0.10.0 (#1 root declaration order, #2 MemoryError with evidence)."""
import copy
import tracemalloc

import numpy as np
import pytest

from fxtensor_bayesnet import Engine, Net, network, run

from conftest import EXAMPLE_CONFIGS, assert_close, chain, load

BASE = {"as_of": "2026-10-11",
        "priors": {"a": {"p": [0.3, 0.7]}, "b": {"p": [0.6, 0.4]}},
        "kernels": {"c": {"parents": ["a"], "cpt": {"x": [0.9, 0.1], "y": [0.2, 0.8]}},
                    "d": {"parents": ["c", "b"], "cpt": {"x|x": [0.5, 0.5], "x|y": [0.1, 0.9],
                                                          "y|x": [0.7, 0.3], "y|y": [0.4, 0.6]}}},
        "observations": [{"var": "d", "label": "y", "focus": ["a", "b"]}]}
L = {"name": "n", "labels": ["x", "y"]}


# --- #1: a root declared after its children ----------------------------------------------
@pytest.mark.parametrize("order", [["a", "b", "c", "d"], ["a", "c", "b", "d"], ["d", "c", "b", "a"], ["c", "d", "a", "b"]])
def test_root_declared_after_children(order):
    res = run(dict(BASE, variables={v: L for v in order}))
    ref = run(dict(BASE, variables={v: L for v in ["a", "b", "c", "d"]}))
    assert res["検算"]["einsum(全同時)との最大差"] < 1e-12
    assert res["シナリオ"]["基本"]["周辺"]["d"] == ref["シナリオ"]["基本"]["周辺"]["d"] == {"x": 0.4816, "y": 0.5184}
    assert res["ベイズ更新(基本)"]["d=y を観測"]["事後"]["a"] == ref["ベイズ更新(基本)"]["d=y を観測"]["事後"]["a"]


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=["weather", "maintenance"])
def test_examples_with_roots_moved_to_the_end(path):
    cfg = load(path)
    moved = copy.deepcopy(cfg)
    roots = [v for v in cfg["variables"] if v in cfg["priors"]]
    moved["variables"] = {**{v: s for v, s in cfg["variables"].items() if v not in roots},
                          **{v: cfg["variables"][v] for v in roots}}
    a, b = run(cfg), run(moved)
    assert_close(a["シナリオ"], b["シナリオ"])
    assert_close(a["ベイズ更新(基本)"], b["ベイズ更新(基本)"])


# --- #2: MemoryError when conditioning with a joint of ~560k-2M elements ----------------------
def _chain_posterior_v0(n, k=3):
    """P(v0 | v{n-1}=s0) in closed form (matrix power of the transition matrix)."""
    row = [0.6] + [0.4 / (k - 1)] * (k - 1)
    T = np.array([row[-j:] + row[:-j] if j else row for j in range(k)])
    lik = np.linalg.matrix_power(T, n - 1)[:, 0]
    p = np.full(k, 1 / k) * lik
    return p / p.sum(), float((np.full(k, 1 / k) @ np.linalg.matrix_power(T, n - 1))[0])


def test_chain_13x3_with_evidence_runs_in_modest_memory():
    cfg = chain(13)          # joint = 3**13 = 1,594,323 elements, "joint" mode
    tracemalloc.start()
    try:
        res = run(cfg)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert res["検算"]["計算モード"] == "joint"
    # 0.9.1 tried to allocate a 14.3M-element (115 MB) wiring kernel and raised MemoryError.
    assert peak < 100 * 2**20, f"peak {peak / 2**20:.0f} MiB"
    post, ev = _chain_posterior_v0(13)
    upd = res["ベイズ更新(基本)"]["v12=s0 を観測"]
    assert upd["観測前の確率"] == pytest.approx(ev, abs=1e-4)
    assert list(upd["事後"]["v0"].values()) == pytest.approx(post, abs=1e-4)
    assert res["検算"]["einsum(条件付け)との最大差"] < 1e-9


def test_chain_10x4_joint_evidence():
    """4^10 = 1,048,576 elements with joint (multi-variable) evidence."""
    cfg = chain(10, k=4)
    cfg["observations"] = [{"evidence": {"v9": "s0", "v5": "s1"}, "focus": ["v0"]}]
    res = run(cfg)
    assert res["検算"]["計算モード"] == "joint"
    assert res["検算"]["einsum(条件付け)との最大差"] < 1e-9


@pytest.mark.parametrize("n", [12, 13, 14])
def test_chain_modes_agree(n):
    a = run(chain(n))
    b = run(chain(n), max_joint=0)            # force the ancestral (variable-elimination) path
    assert b["検算"]["計算モード"] == "ancestral"
    assert_close(a["ベイズ更新(基本)"], b["ベイズ更新(基本)"])
    assert_close(a["シナリオ"], b["シナリオ"])


def test_wiring_fallback_matches_library_path(monkeypatch, example_cfg):
    """When the dense wiring kernel would be too large, marginals/conditioning fall back to einsum; same numbers."""
    ref = run(example_cfg)
    monkeypatch.setattr(network, "MAX_WIRING", 200)   # above the tiny self-test, below the examples
    res = run(example_cfg)
    for sc in ref["シナリオ"]:
        for v, d in ref["シナリオ"][sc]["周辺"].items():
            assert list(res["シナリオ"][sc]["周辺"][v].values()) == pytest.approx(list(d.values()), abs=1e-4)
    for k, u in ref["ベイズ更新(基本)"].items():
        for w, d in u["事後"].items():
            assert list(res["ベイズ更新(基本)"][k]["事後"][w].values()) == pytest.approx(list(d.values()), abs=1e-4)


def test_engine_marginal_uses_wiring_when_small():
    cfg = dict(BASE, variables={v: L for v in "abcd"})
    eng = Engine(Net(cfg), {"a": [0.3, 0.7], "b": [0.6, 0.4]})
    assert eng.mode == "joint" and eng._wiring_fits(["a", "d"])


def test_version_is_consistent():
    import os
    import re

    import fxtensor_bayesnet
    from conftest import ROOT
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        v = re.search(r'^version = "([^"]+)"', f.read(), re.M).group(1)
    assert v == fxtensor_bayesnet.__version__ == "0.11.1"
    with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as f:
        assert f"## [{v}]" in f.read()
