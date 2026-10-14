"""Regression tests for 0.11.1 (#24): two effects with the same (treatment, outcome, outcome_label) used the same
result key, so the first one was silently overwritten."""
import copy
import json

import pytest

from fxtensor_bayesnet import run, to_english_keys
from fxtensor_bayesnet.cli import main
from fxtensor_bayesnet.config import ConfigError, validate
from fxtensor_bayesnet.report import render_markdown

L2 = ["0", "1"]
BASE = {
    "as_of": "2026-10-15",
    "variables": {"z": {"labels": L2}, "x": {"labels": L2}, "m": {"labels": L2}, "y": {"labels": L2}},
    "priors": {"z": {"p": [0.5, 0.5]}},
    "kernels": {"x": {"parents": ["z"], "cpt": {"0": [0.7, 0.3], "1": [0.3, 0.7]}},
                "m": {"parents": ["x", "z"], "cpt": {"0|0": [0.8, 0.2], "0|1": [0.6, 0.4],
                                                     "1|0": [0.4, 0.6], "1|1": [0.1, 0.9]}},
                "y": {"parents": ["m"], "cpt": {"0": [0.7, 0.3], "1": [0.2, 0.8]}}},
}


def cfg_with(*effects):
    c = copy.deepcopy(BASE)
    c["effects"] = list(effects)
    return c


E = {"treatment": "x", "outcome": "y", "outcome_label": "1"}


def test_strata_are_part_of_the_key():
    cfg = cfg_with({**E, "given_do": {"z": None}}, {**E, "given_do": {"m": None}}, {**E, "given": {"z": None}},
                   {**E, "given_do": {"m": None}, "given": {"z": None}}, dict(E))
    validate(cfg)
    ce = run(cfg)["因果効果"]
    assert list(ce) == ["x → y=1 | do(z)", "x → y=1 | do(m)", "x → y=1 | z", "x → y=1 | do(m), z", "x → y=1"]
    assert ce["x → y=1 | do(z)"]["層別ACE(given_do)"]["条件"] == ["z"]
    assert ce["x → y=1 | do(m)"]["層別ACE(given_do)"]["条件"] == ["m"]
    assert "名前" not in ce["x → y=1"]      # plain effects keep the 0.11.0 key and fields


def test_name_is_used_as_key():
    cfg = cfg_with({**E, "name": "by z", "given_do": {"z": None}}, {**E, "name": "by m", "given_do": {"m": None}})
    res = run(cfg)
    assert list(res["因果効果"]) == ["by z", "by m"] and res["因果効果"]["by z"]["名前"] == "by z"
    en = to_english_keys(res)["effects"]
    assert list(en) == ["by z", "by m"] and en["by m"]["name"] == "by m"
    for lang in ("ja", "en"):
        md = render_markdown(res, cfg, lang=lang)
        assert "### by z" in md and "### by m" in md


@pytest.mark.parametrize("effects", [
    [dict(E), dict(E)],
    [{**E, "given_do": {"z": ["0"]}}, {**E, "given_do": {"z": ["1"]}}],
    [{**E, "name": "a"}, {"treatment": "z", "outcome": "y", "name": "a"}],
    [{**E, "name": "x → y=1"}, dict(E)],
])
def test_colliding_keys_are_errors(effects, tmp_path, capsys):
    cfg = cfg_with(*effects)
    with pytest.raises(ConfigError, match="重なる"):
        validate(cfg)
    with pytest.raises(ConfigError, match="重なる"):
        run(cfg)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    assert main(["--config", str(p), "--check"]) == 2


@pytest.mark.parametrize("name", ["", "  ", 3])
def test_bad_name(name):
    with pytest.raises(ConfigError, match="name"):
        validate(cfg_with({**E, "name": name}))


def test_plot_titles_distinguish_effects(monkeypatch):
    pytest.importorskip("matplotlib")
    from fxtensor_bayesnet import plot
    cfg = cfg_with({**E, "given_do": {"z": None}}, {**E, "name": "via m", "given_do": {"m": None}})
    res = run(cfg)
    seen = {}

    def fake_save(fig, path, dpi):
        seen["titles"] = [ax.get_title() for ax in fig.axes if ax.axison]
        plot._mpl().close(fig)
        return path

    monkeypatch.setattr(plot, "_save", fake_save)
    plot.plot_effects(res, cfg, "unused.png")
    assert len(seen["titles"]) == 4
    assert seen["titles"][0].startswith("x → y=1 | do(z)\n") and seen["titles"][2].startswith("via m\n")
    assert seen["titles"][3].startswith("via m\n")
