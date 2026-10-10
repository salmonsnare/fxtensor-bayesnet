import json
import os
import shutil

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
EXAMPLE_CONFIGS = [os.path.join(EXAMPLES, d, "config.json") for d in ("weather", "maintenance")]


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
