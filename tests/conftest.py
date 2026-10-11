import json
import os
import shutil

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
EXAMPLE_CONFIGS = [os.path.join(EXAMPLES, d, "config.json") for d in ("weather", "maintenance")]
ECOLOGY_DIR = os.path.join(EXAMPLES, "ecology_host_parasitoid")
ECOLOGY_CONFIG = os.path.join(ECOLOGY_DIR, "config.json")


def chain(n, k=3, observe=True):
    """v0 → v1 → … → v{n-1}（各 k 値）。v{n-1}=s0 を観測。"""
    labs = [f"s{i}" for i in range(k)]
    V = {f"v{i}": {"name": f"v{i}", "labels": labs} for i in range(n)}
    row = [0.6] + [0.4 / (k - 1)] * (k - 1)
    K = {f"v{i}": {"parents": [f"v{i-1}"], "cpt": {lab: row[-j:] + row[:-j] if j else row for j, lab in enumerate(labs)}}
         for i in range(1, n)}
    cfg = {"as_of": "2026-10-11", "variables": V, "priors": {"v0": {"p": [1 / k] * k}}, "kernels": K,
           "outputs": [f"v{n-1}"]}
    if observe:
        cfg["observations"] = [{"var": f"v{n-1}", "label": "s0", "focus": ["v0"]}]
    return cfg


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(params=EXAMPLE_CONFIGS, ids=["weather", "maintenance"])
def example_cfg(request):
    return load(request.param)


@pytest.fixture
def tmp_config(tmp_path):
    """天気の例を一時フォルダへコピーし、そのパスを返す（examples/ を汚さない）。"""
    dst = tmp_path / "config.json"
    shutil.copy(EXAMPLE_CONFIGS[0], dst)
    return dst


def assert_close(a, b, tol=1.5e-4, path="res"):
    """Compare two result trees; numbers may differ by one unit in the 4th decimal (rounding at a .5 boundary
    can go either way when two exact computation paths differ by ~1e-16)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{path}: keys {list(a)} != {list(b)}"
        for k in a:
            assert_close(a[k], b[k], tol, f"{path}[{k!r}]")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, tol, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= tol, f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"
