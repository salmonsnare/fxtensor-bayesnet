"""設定の検証（--check）と整形。"""
import copy
import json
import subprocess
import sys

import pytest

from fxtensor_bayesnet import ConfigError, dump_config, validate
from fxtensor_bayesnet.cli import main

from conftest import EXAMPLE_CONFIGS, load


def test_examples_are_valid(example_cfg):
    validate(example_cfg)


def broken(cfg, fn):
    c = copy.deepcopy(cfg)
    fn(c)
    return c


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c["priors"]["pressure"].update(p=[0.5, 0.3, 0.3]), "和が1でない"),
    (lambda c: c["priors"]["pressure"].update(p=[0.5, 0.5]), "長さ 3"),
    (lambda c: c["priors"]["humidity"].update(p=[1.2, -0.2, 0.0]), "負の確率"),
    (lambda c: c["kernels"]["cloud"]["cpt"].pop("高気圧|低"), "行が足りない"),
    (lambda c: c["kernels"]["event"]["cpt"].update({"雪": [1, 0, 0]}), "余分な行"),
    (lambda c: c["kernels"]["event"].update(parents=["snow"]), "未定義の親"),
    (lambda c: c["kernels"]["rain"].update(parents=["cloud", "pressure", "event"]), "行が足りない"),
    (lambda c: c["scenarios"].update({"X": {"rain": [1, 0, 0]}}), "根の変数ではない"),
    (lambda c: c["observations"].append({"var": "rain", "label": "雪"}), "不正な観測"),
    (lambda c: c["outputs"].append("snow"), "outputs"),
    (lambda c: c["variables"]["cloud"].update(labels=["晴れ"]), "labels が不正"),
    (lambda c: c["priors"].update(cloud={"p": [1, 0, 0]}), "どちらか一方"),
])
def test_validate_rejects(mutate, msg):
    cfg = load(EXAMPLE_CONFIGS[0])   # 天気の例を壊して検査
    with pytest.raises(ConfigError, match=msg):
        validate(broken(cfg, mutate))


def test_cycle_detected():
    cfg = {
        "as_of": "2026-01-01",
        "variables": {"R": {"labels": ["x", "y"]}, "A": {"labels": ["x", "y"]}, "B": {"labels": ["x", "y"]}},
        "priors": {"R": {"p": [0.5, 0.5]}},
        "kernels": {
            "A": {"parents": ["B"], "cpt": {"x": [1, 0], "y": [0, 1]}},
            "B": {"parents": ["A"], "cpt": {"x": [1, 0], "y": [0, 1]}},
        },
    }
    with pytest.raises(ConfigError, match="循環"):
        validate(cfg)


def test_cli_check_ok(tmp_config, capsys):
    assert main(["--config", str(tmp_config), "--check"]) == 0
    assert "設定OK" in capsys.readouterr().out
    assert not (tmp_config.parent / "results.json").exists()


def test_cli_check_reports_all_errors(tmp_config, capsys):
    cfg = json.loads(tmp_config.read_text(encoding="utf-8"))
    cfg["priors"]["pressure"]["p"] = [0.5, 0.3, 0.3]
    del cfg["kernels"]["cloud"]["cpt"]["高気圧|低"]
    tmp_config.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    assert main(["--config", str(tmp_config), "--check"]) == 2
    err = capsys.readouterr().err
    assert "和が1でない" in err and "行が足りない" in err


def test_cli_check_bad_json_and_missing(tmp_path, capsys):
    p = tmp_path / "config.json"
    p.write_text("{ broken", encoding="utf-8")
    assert main(["--config", str(p), "--check"]) == 2
    assert main(["--config", str(tmp_path / "none.json"), "--check"]) == 2


def test_console_script_check_exit_codes(tmp_config):
    """インストールされた CLI（python -m 経由）の終了コード。"""
    r = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--config", str(tmp_config), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cfg = json.loads(tmp_config.read_text(encoding="utf-8"))
    cfg["priors"]["humidity"]["p"] = [0.9, 0.9, 0.9]
    tmp_config.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--config", str(tmp_config), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "和が1でない" in r.stderr


def test_format_config_roundtrip(tmp_config):
    before = json.loads(tmp_config.read_text(encoding="utf-8"))
    ugly = json.dumps(before, ensure_ascii=False, indent=8)
    tmp_config.write_text(ugly, encoding="utf-8")
    assert main(["--config", str(tmp_config), "--format-config", "--check"]) == 0
    text = tmp_config.read_text(encoding="utf-8")
    assert json.loads(text) == before
    assert text == dump_config(before)
    assert '"高気圧|低": [0.8, 0.17, 0.03]' in text     # 核の1行が1行になる
