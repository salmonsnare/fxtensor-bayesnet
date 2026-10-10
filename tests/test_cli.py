"""CLI：--no-write / --md / 出力ファイル / --snapshot / --diff。"""
import json

from fxtensor_bayesnet.cli import main


def test_no_write_prints_json_and_writes_nothing(tmp_config, capsys):
    assert main(["--config", str(tmp_config), "--no-write"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert set(res["シナリオ"]) == {"基本", "A:台風接近", "B:乾燥した晴天続き"}
    assert sorted(p.name for p in tmp_config.parent.iterdir()) == ["config.json"]


def test_md_output(tmp_config, capsys):
    assert main(["--config", str(tmp_config), "--no-write", "--md"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# 結果（基準日 2026-10-10")
    assert "## rain（降水）" in out and "## ベイズ更新（基本）" in out


def test_writes_results_next_to_config(tmp_config, capsys):
    assert main(["--config", str(tmp_config)]) == 0
    d = tmp_config.parent
    assert json.loads((d / "results.json").read_text(encoding="utf-8"))["基準日"] == "2026-10-10"
    assert (d / "results.md").read_text(encoding="utf-8").startswith("# 結果")
    mmd = (d / "model.mmd").read_text(encoding="utf-8")
    assert mmd.startswith("graph TD") and "pressure --> cloud" in mmd


def test_out_dir(tmp_config, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["--config", str(tmp_config), "--out-dir", str(out)]) == 0
    assert (out / "results.json").exists() and not (tmp_config.parent / "results.json").exists()


def test_snapshot_force_and_diff(tmp_config, capsys):
    d = tmp_config.parent
    assert main(["--config", str(tmp_config), "--no-write", "--snapshot"]) == 0
    snap = d / "history" / "2026-10-10"
    assert (snap / "config.json").exists() and (snap / "results.json").exists()
    # 同じ日付は --force なしでは上書きしない
    assert main(["--config", str(tmp_config), "--no-write", "--snapshot"]) == 1
    assert main(["--config", str(tmp_config), "--no-write", "--snapshot", "--force"]) == 0
    capsys.readouterr()

    # 翌日：事前分布と核の1行を変えて差分を見る
    cfg = json.loads(tmp_config.read_text(encoding="utf-8"))
    cfg["as_of"] = "2026-10-11"
    cfg["priors"]["pressure"]["p"] = [0.2, 0.3, 0.5]
    cfg["kernels"]["traffic"]["cpt"]["大雨"] = [0.1, 0.3, 0.6]
    tmp_config.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    assert main(["--config", str(tmp_config), "--diff"]) == 0
    out = capsys.readouterr().out
    assert "history/2026-10-10/config.json" in out
    assert "~ 事前分布 pressure: [0.45, 0.3, 0.25] → [0.2, 0.3, 0.5]" in out
    assert "~ 核 traffic[大雨]: [0.15, 0.35, 0.5] → [0.1, 0.3, 0.6]" in out
    assert "# 結果の差分" in out and "~ 基本 / rain:" in out

    # パスを明示した比較
    assert main(["--config", str(tmp_config), "--diff", str(snap / "config.json")]) == 0
    assert "~ 事前分布 pressure" in capsys.readouterr().out


def test_diff_without_history(tmp_config, capsys):
    assert main(["--config", str(tmp_config), "--diff"]) == 0
    assert "比較対象のスナップショットがありません" in capsys.readouterr().out


def test_diff_no_change(tmp_config, capsys):
    main(["--config", str(tmp_config), "--no-write", "--snapshot"])
    cfg = json.loads(tmp_config.read_text(encoding="utf-8"))
    cfg["as_of"] = "2026-10-11"
    tmp_config.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    capsys.readouterr()
    main(["--config", str(tmp_config), "--diff"])
    out = capsys.readouterr().out
    assert "（設定の変更なし）" in out and "（結果の変化なし）" in out
