"""Regression tests for the 0.9.1 fixes."""
import copy
import json
import math
import os
import subprocess
import sys

import pytest

from fxtensor_bayesnet import run
from fxtensor_bayesnet.cli import main
from fxtensor_bayesnet.history import diff_configs, diff_results
from fxtensor_bayesnet.inference import movement
from fxtensor_bayesnet.report import render_markdown

from conftest import EXAMPLE_CONFIGS, load


def next_day(tmp_config, **prior_overrides):
    cfg = json.loads(tmp_config.read_text(encoding="utf-8"))
    cfg["as_of"] = "2026-10-11"
    for v, p in prior_overrides.items():
        cfg["priors"][v]["p"] = p
    tmp_config.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")


# 1. --diff with a folder / missing path -------------------------------------------------
def test_diff_accepts_snapshot_folder(tmp_config, capsys):
    assert main(["--config", str(tmp_config), "--no-write", "--snapshot"]) == 0
    next_day(tmp_config, pressure=[0.2, 0.3, 0.5])
    capsys.readouterr()
    folder = tmp_config.parent / "history" / "2026-10-10"
    assert main(["--config", str(tmp_config), "--diff", str(folder)]) == 0
    out = capsys.readouterr().out
    assert f"# 設定の差分: {folder / 'config.json'}" in out
    assert "~ 事前分布 pressure" in out and "# 結果の差分" in out


def test_diff_folder_with_trailing_slash_and_relative(tmp_config, capsys, monkeypatch):
    main(["--config", str(tmp_config), "--no-write", "--snapshot"])
    next_day(tmp_config, pressure=[0.2, 0.3, 0.5])
    monkeypatch.chdir(tmp_config.parent)
    capsys.readouterr()
    assert main(["--config", "config.json", "--diff", "history/2026-10-10/"]) == 0
    assert "# 結果の差分" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["no/such/path", "no_such.json"])
def test_diff_missing_path_exit_2(tmp_config, capsys, target):
    assert main(["--config", str(tmp_config), "--diff", str(tmp_config.parent / target)]) == 2
    assert "比較対象が見つかりません" in capsys.readouterr().err


def test_diff_folder_without_config_exit_2(tmp_config, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--config", str(tmp_config), "--diff", str(empty)]) == 2
    assert "config.json がありません" in capsys.readouterr().err


def test_diff_folder_no_traceback_subprocess(tmp_config):
    main(["--config", str(tmp_config), "--no-write", "--snapshot"])
    folder = tmp_config.parent / "history" / "2026-10-10"
    r = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--config", str(tmp_config), "--diff", str(folder)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "Traceback" not in r.stderr
    r = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--config", str(tmp_config), "--diff",
                        str(tmp_config.parent / "nope")], capture_output=True, text=True)
    assert r.returncode == 2 and "Traceback" not in r.stderr


def test_diff_same_date_is_not_a_target(tmp_config, capsys):
    main(["--config", str(tmp_config), "--no-write", "--snapshot"])     # history/2026-10-10 == as_of
    capsys.readouterr()
    assert main(["--config", str(tmp_config), "--diff"]) == 0
    out = capsys.readouterr().out
    assert "比較対象のスナップショットがありません" in out and "2026-10-10 は対象外" in out


# 2. --snapshot collision is checked before anything is written ---------------------------
def test_snapshot_collision_writes_nothing(tmp_config, tmp_path, capsys):
    hist = tmp_path / "hist"
    assert main(["--config", str(tmp_config), "--no-write", "--snapshot", "--history-dir", str(hist)]) == 0
    snap = hist / "2026-10-10" / "results.json"
    before = snap.read_bytes()
    mtime = os.stat(snap).st_mtime_ns
    out = tmp_path / "out"
    rc = main(["--config", str(tmp_config), "--snapshot", "--history-dir", str(hist), "--out-dir", str(out)])
    assert rc == 1
    assert "何も書かずに終了" in capsys.readouterr().err
    assert not out.exists()                          # results / figures not written
    assert snap.read_bytes() == before and os.stat(snap).st_mtime_ns == mtime
    # --force still works and writes everything
    assert main(["--config", str(tmp_config), "--snapshot", "--force", "--history-dir", str(hist),
                 "--out-dir", str(out)]) == 0
    assert (out / "results.json").exists()


def test_snapshot_collision_with_plot_writes_no_figures(tmp_config, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    main(["--config", str(tmp_config), "--no-write", "--snapshot"])
    out = tmp_path / "figs"
    assert main(["--config", str(tmp_config), "--snapshot", "--plot", "--out-dir", str(out)]) == 1
    assert not out.exists()


# 3. --diff shows prior reasons and rows of re-wired kernels -------------------------------
def test_diff_shows_prior_reason_change():
    old = load(EXAMPLE_CONFIGS[0])
    new = copy.deepcopy(old)
    new["priors"]["pressure"]["reason"] = "updated rationale"
    out = diff_configs(old, new)
    assert any(line.startswith("~ 理由文 pressure（事前分布）を更新") and "updated rationale" in line for line in out)


def test_diff_shows_rows_when_parents_change():
    old = load(EXAMPLE_CONFIGS[0])
    new = copy.deepcopy(old)
    new["kernels"]["traffic"]["parents"] = ["rain", "pressure"]
    new["kernels"]["traffic"]["cpt"] = {f"{r}|{p}": [0.5, 0.3, 0.2]
                                        for r in ["なし", "小雨", "大雨"] for p in ["高気圧", "前線停滞", "低気圧接近"]}
    out = diff_configs(old, new)
    assert "~ 経路変更 traffic: 親 ['rain'] → ['rain', 'pressure']（核の行を全て置き換え）" in out
    assert "  - 旧 traffic[大雨]: [0.15, 0.35, 0.5]" in out
    assert "  + 新 traffic[大雨|低気圧接近]: [0.5, 0.3, 0.2]" in out
    assert sum(line.startswith("  - 旧 traffic[") for line in out) == 3
    assert sum(line.startswith("  + 新 traffic[") for line in out) == 9


def test_diff_shows_added_and_removed_rows_on_label_change():
    old = load(EXAMPLE_CONFIGS[0])
    new = copy.deepcopy(old)
    new["variables"]["rain"]["labels"] = ["なし", "小雨", "豪雨"]
    for k in ("event", "traffic"):
        new["kernels"][k]["cpt"]["豪雨"] = new["kernels"][k]["cpt"].pop("大雨")
    out = diff_configs(old, new)
    assert "- 核の行削除 event[大雨]: [0.1, 0.3, 0.6]" in out
    assert "+ 核の行追加 event[豪雨]: [0.1, 0.3, 0.6]" in out


# 4. checks compare unrounded values --------------------------------------------------------
@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=["weather", "maintenance"])
def test_conditioning_check_uses_unrounded_values(path):
    chk = run(load(path))["検算"]
    # rounding to 4 decimals used to give ~1e-5; unrounded comparison is at machine precision
    assert chk["einsum(条件付け)との最大差"] < 1e-12
    assert chk["bayesian_inversionとconditionalizationの最大差"] < 1e-12


# 5. no "-0.0" -------------------------------------------------------------------------------
def test_diff_results_has_no_negative_zero():
    old = {"シナリオ": {"基本": {"周辺": {"X": {"x": 0.3, "y": 0.3004, "z": 0.3996}}}}}
    new = {"シナリオ": {"基本": {"周辺": {"X": {"x": 0.302, "y": 0.3, "z": 0.398}}}}}
    out = "\n".join(diff_results(old, new))
    assert "y +0.0pt" in out and "-0.0" not in out


def test_movement_has_no_negative_zero():
    mv = movement([0.5, 0.5], {"a": 0.5000001, "b": 0.4999999}, ["a", "b"])
    for x in list(mv["差(pt)"].values()) + [mv["全変動距離TV"], mv["KL(事後||事前)[nat]"]]:
        assert x == 0 and math.copysign(1.0, x) == 1.0


def test_markdown_has_no_negative_zero():
    """B is independent of the observation; its posterior equals the prior up to float noise,
    which used to be printed as "-0.0" (KL and 差)."""
    cfg = {"as_of": "2026-01-01",
           "variables": {"A": {"labels": ["a0", "a1"]}, "B": {"labels": ["b0", "b1", "b2"]},
                         "C": {"labels": ["c0", "c1"]}},
           "priors": {"A": {"p": [0.3, 0.7]}, "B": {"p": [0.3, 0.45, 0.25]}},
           "kernels": {"C": {"parents": ["A"], "cpt": {"a0": [0.9, 0.1], "a1": [0.2, 0.8]}}},
           "observations": [{"var": "C", "label": "c1", "focus": ["A", "B"]}], "outputs": ["C"]}
    for lang in ("ja", "en"):
        md = render_markdown(run(cfg), cfg, lang=lang)
        assert "-0.0" not in md
        assert "KL 0.0" in md
