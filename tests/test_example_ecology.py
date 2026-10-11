"""The bundled ecology example (#8) runs, validates and is documented as illustrative."""
import os
import runpy
import sys

import pytest

from fxtensor_bayesnet import load_config, run
from fxtensor_bayesnet.cli import main

from conftest import ECOLOGY_CONFIG, ECOLOGY_DIR


def test_config_is_valid_and_runs():
    res = run(load_config(ECOLOGY_CONFIG))
    assert {"介入", "因果効果", "感度分析"} <= set(res)
    assert len(res["ベイズ更新(基本)"]) == 4


def test_cli_check():
    assert main(["--config", ECOLOGY_CONFIG, "--check"]) == 0


def test_script_runs(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_example.py"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(os.path.join(ECOLOGY_DIR, "run_example.py"), run_name="__main__")
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "illustrative placeholders" in out and "ACE +2.9pt" in out


def test_readme_states_numbers_are_illustrative():
    with open(os.path.join(ECOLOGY_DIR, "README.md"), encoding="utf-8") as f:
        text = f.read()
    assert "illustrative placeholder" in text.lower()
    assert "https://doi.org/10.64898/2026.10.06.757064" in text
    assert "not estimates from that paper" in text.lower()
