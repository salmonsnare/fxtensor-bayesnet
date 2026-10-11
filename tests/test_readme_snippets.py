"""The config and Python snippets in README.md / README-ja.md run (keeps the documentation honest)."""
import json
import os
import re

import pytest

from fxtensor_bayesnet import load_config, run, validate

from conftest import ROOT

CASES = [("README.md", "### Interventions, causal effects, evidence and sensitivity", "config.en.json", "## Python API"),
         ("README-ja.md", "### 介入・因果効果・証拠・感度分析", "config.json", "## Python から使う")]


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("readme,section,cfgname,_", CASES, ids=["en", "ja"])
def test_config_snippet(readme, section, cfgname, _):
    text = read(readme)
    block = re.search(r"```json\n(.*?)```", text[text.index(section):], re.S).group(1)
    cfg = load_config(os.path.join(ROOT, "examples", "maintenance", cfgname))
    cfg.update(json.loads("{" + block + "}"))
    validate(cfg)
    res = run(cfg)
    e = next(iter(res["因果効果"].values()))
    assert (round(e["観測上の差"] * 100, 1), round(e["ACE"] * 100, 1)) == (14.9, 5.8)     # numbers quoted in the text
    assert f"{e['観測上の差'] * 100:+.1f}pt" in text[text.index(section):]


@pytest.mark.parametrize("readme,_,__,section", CASES, ids=["en", "ja"])
def test_python_snippet(readme, _, __, section, monkeypatch):
    text = read(readme)
    code = re.search(r"```python\n(.*?)```", text[text.index(section):], re.S).group(1)
    code = code[:code.index("# 図") if "# 図" in code else code.index("# figures")]
    monkeypatch.chdir(ROOT)
    ns = {}
    exec(code, ns)
    assert 0 < ns["p_e"] < 1 and "cloud" in ns["surgered"]["priors"]
