"""English keys in results JSON (#6)."""
import copy
import json

from fxtensor_bayesnet import load_config, run, to_english_keys
from fxtensor_bayesnet.cli import main
from fxtensor_bayesnet.keys import KEY_MAP

from conftest import ECOLOGY_CONFIG


def test_english_keys_cover_all_fixed_keys():
    res = to_english_keys(run(load_config(ECOLOGY_CONFIG)))
    assert set(res) == {"as_of", "horizon", "note", "config_hash", "priors_base", "scenarios", "updates_base",
                        "interventions", "effects", "sensitivity", "checks"}
    assert "base" in res["scenarios"] and set(res["scenarios"]["base"]) == {"overrides", "marginals"}
    u = res["updates_base"]["fledge=Low"]
    assert set(u) == {"evidence_probability", "posterior", "focus_prior", "focus_movement", "ranking_by_tv"}
    assert set(u["focus_movement"]["agri"]) == {"delta_pt", "tv_distance", "kl_posterior_prior_nat"}
    assert res["updates_base"]["fly=High, weather=Poor"]["evidence"] == {"fly": "High", "weather": "Poor"}
    soft = res["updates_base"]["Fly count suggests High (imperfect count)"]["evidence"]
    assert soft == {"fly": {"likelihood": [0.2, 0.9]}}
    e = res["effects"]["parasitoid → fledge=High"]
    assert {"ace", "observed_difference", "confounding_gap", "p_outcome_do", "p_outcome_observed"} <= set(e)
    assert set(res["sensitivity"]["fledge=High"]["top"][0]) == {
        "parameter", "base_value", "low_value", "high_value", "p_low", "p_high", "swing_pt"}
    assert "max_err_intervention_vs_truncated_factorization" in res["checks"]
    # no Japanese key left anywhere (values like the remark text may stay Japanese)
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    assert not [k for k in keys(res) if k in KEY_MAP]


def test_user_names_are_not_translated():
    """A label that happens to equal a fixed Japanese key must stay as is."""
    cfg = {"as_of": "2026-01-01",
           "variables": {"x": {"name": "x", "labels": ["事後", "周辺"]}, "y": {"name": "y", "labels": ["基本", "注意"]}},
           "priors": {"x": {"p": [0.5, 0.5]}},
           "kernels": {"y": {"parents": ["x"], "cpt": {"事後": [0.9, 0.1], "周辺": [0.2, 0.8]}}},
           "observations": [{"var": "y", "label": "注意"}]}
    res = to_english_keys(run(cfg))
    assert set(res["scenarios"]["base"]["marginals"]["y"]) == {"基本", "注意"}
    assert set(res["updates_base"]["y=注意"]["posterior"]["x"]) == {"事後", "周辺"}


def test_cli_json_keys_en_and_snapshot_stays_canonical(tmp_config, capsys):
    assert main(["--config", str(tmp_config), "--json-keys", "en", "--snapshot"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "scenarios" in out and "シナリオ" not in out
    written = json.loads((tmp_config.parent / "results.json").read_text(encoding="utf-8"))
    assert "scenarios" in written
    snap = json.loads((tmp_config.parent / "history" / "2026-10-10" / "results.json").read_text(encoding="utf-8"))
    assert "シナリオ" in snap          # --diff keeps working across versions
    cfg = json.loads(tmp_config.read_text(encoding="utf-8"))
    cfg["as_of"] = "2026-10-11"
    tmp_config.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    assert main(["--config", str(tmp_config), "--diff"]) == 0
    assert "（結果の変化なし）" in capsys.readouterr().out


def test_default_keys_unchanged(tmp_config, capsys):
    assert main(["--config", str(tmp_config), "--no-write"]) == 0
    assert "シナリオ" in json.loads(capsys.readouterr().out)


def test_to_english_keys_does_not_mutate():
    res = run(load_config(ECOLOGY_CONFIG))
    before = copy.deepcopy(res)
    to_english_keys(res)
    assert res == before
