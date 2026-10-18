"""Regression tests for 0.11.3 (#33): negative stratum-ACE labels overlapped the overall-ACE note."""
import re

import pytest

from fxtensor_bayesnet import run
from fxtensor_bayesnet.i18n import t

# Issue #33: strata -15 / -23 pt, overall -19 pt. Labels used to fall below the bars.
NEG = {
    "as_of": "2026-10-19",
    "variables": {"x": {"labels": ["a", "b"]}, "z": {"labels": ["L", "H"]}, "y": {"labels": ["n", "p"]}},
    "priors": {"x": {"p": [0.5, 0.5]}, "z": {"p": [0.5, 0.5]}},
    "kernels": {"y": {"parents": ["x", "z"], "cpt": {
        "a|L": [0.60, 0.40], "b|L": [0.75, 0.25], "a|H": [0.50, 0.50], "b|H": [0.73, 0.27]}}},
    "effects": [{"treatment": "x", "outcome": "y", "outcome_label": "p", "given_do": {"z": None}}],
}

# One bar long enough to hold its label, one too short (placed on the 0 side).
SHORT = {
    "as_of": "2026-10-19",
    "variables": {"x": {"labels": ["a", "b"]}, "z": {"labels": ["L", "H"]}, "y": {"labels": ["n", "p"]}},
    "priors": {"x": {"p": [0.5, 0.5]}, "z": {"p": [0.5, 0.5]}},
    "kernels": {"y": {"parents": ["x", "z"], "cpt": {
        "a|L": [0.50, 0.50], "b|L": [0.70, 0.30], "a|H": [0.50, 0.50], "b|H": [0.501, 0.499]}}},
    "effects": [{"treatment": "x", "outcome": "y", "outcome_label": "p", "given_do": {"z": None}}],
}

POS = {
    "as_of": "2026-10-19",
    "variables": {"x": {"labels": ["a", "b"]}, "z": {"labels": ["L", "H"]}, "y": {"labels": ["n", "p"]}},
    "priors": {"x": {"p": [0.5, 0.5]}, "z": {"p": [0.5, 0.5]}},
    "kernels": {"y": {"parents": ["x", "z"], "cpt": {
        "a|L": [0.70, 0.30], "b|L": [0.20, 0.80], "a|H": [0.60, 0.40], "b|H": [0.30, 0.70]}}},
    "effects": [{"treatment": "x", "outcome": "y", "outcome_label": "p", "given_do": {"z": None}}],
}

_VALUE = re.compile(r"^[+-]\d+\.\d$")


def _overlap(a, b, tol=0.5):
    """True when two display bboxes share more than tol pixels on both axes."""
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol and
            min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def _inside(bb, axes, tol=1.5):
    return (bb.x0 >= axes.x0 - tol and bb.x1 <= axes.x1 + tol and
            bb.y0 >= axes.y0 - tol and bb.y1 <= axes.y1 + tol)


@pytest.fixture
def render_effects(monkeypatch):
    pytest.importorskip("matplotlib")
    from fxtensor_bayesnet import plot
    held = {}

    def fake_save(fig, path, dpi):
        fig.canvas.draw()
        held["fig"] = fig
        return path

    monkeypatch.setattr(plot, "_save", fake_save)

    def render(cfg, lang="ja"):
        res = run(cfg)
        assert plot.plot_effects(res, cfg, "unused.png", lang=lang) == "unused.png"
        fig = held["fig"]
        fig.canvas.draw()
        return fig, fig.canvas.get_renderer(), res

    yield render
    if held.get("fig") is not None:
        plot._mpl().close(held["fig"])


def _strata_axes(fig):
    return [ax for ax in fig.axes if ax.get_legend() is not None]


def _value_labels(ax):
    return [tx for tx in ax.texts if _VALUE.match(tx.get_text())]


def _note(ax, lang, ace):
    expected = t(lang, "overall_ace", ace=f"{ace * 100:+.1f}")
    texts = ax.get_legend().get_texts()
    assert [tx.get_text() for tx in texts] == [expected]
    return texts[0]


def _pair(ax, labels):
    """Bar patch → value label, matched on the bar's x centre."""
    out = []
    for bar in ax.patches:
        if bar.get_width() <= 0:
            continue
        cx = bar.get_x() + bar.get_width() / 2
        label = min(labels, key=lambda tx: abs(tx.get_position()[0] - cx))
        out.append((bar, label))
    return out


def _bar_ys(bar, renderer):
    bb = bar.get_window_extent(renderer)
    return min(bb.y0, bb.y1), max(bb.y0, bb.y1)


def test_issue_example_ace_values():
    e = run(NEG)["因果効果"]["x → y=p | do(z)"]
    assert e["ACE"] == pytest.approx(-0.19)
    layers = e["層別ACE(given_do)"]["層"]
    assert layers["z=L"]["ACE"] == pytest.approx(-0.15)
    assert layers["z=H"]["ACE"] == pytest.approx(-0.23)


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_negative_strata_labels_stay_clear(render_effects, lang):
    fig, renderer, res = render_effects(NEG, lang)
    e = res["因果効果"]["x → y=p | do(z)"]
    axes = _strata_axes(fig)
    assert len(axes) == 1
    ax = axes[0]
    note = _note(ax, lang, e["ACE"])
    note_bb = note.get_window_extent(renderer)
    axes_bb = ax.get_window_extent(renderer)
    # The overall ACE is the dashed line's legend entry, outside the axes.
    assert note_bb.y0 >= axes_bb.y1 - 1.0
    handles = getattr(ax.get_legend(), "legend_handles", None) or ax.get_legend().legendHandles
    assert handles[0].get_linestyle() in ("--", "dashed")
    # Not the old corner annotation.
    assert not any("全体の ACE" in tx.get_text() or "overall ACE" in tx.get_text() for tx in ax.texts)
    title_bb = ax.title.get_window_extent(renderer)
    assert not _overlap(title_bb, note_bb)

    labels = _value_labels(ax)
    assert sorted(tx.get_text() for tx in labels) == ["-15.0", "-23.0"]
    zero_y = ax.transData.transform((0.0, 0.0))[1]
    for bar, label in _pair(ax, labels):
        assert bar.get_height() < 0
        bb = label.get_window_extent(renderer)
        lo, _hi = _bar_ys(bar, renderer)
        assert not _overlap(bb, note_bb)
        assert _inside(bb, axes_bb)
        assert bb.y0 >= lo - 1.0          # not below the bar's end
        assert bb.y1 <= zero_y + 1.0      # inside the bar, on the side toward 0
        face = label.get_bbox_patch().get_facecolor()
        assert face[0] == pytest.approx(1) and face[1] == pytest.approx(1) and face[2] == pytest.approx(1)


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_short_negative_bar_label_goes_above_zero(render_effects, lang):
    fig, renderer, res = render_effects(SHORT, lang)
    e = res["因果効果"]["x → y=p | do(z)"]
    layers = e["層別ACE(given_do)"]["層"]
    assert layers["z=L"]["ACE"] == pytest.approx(-0.2)
    assert layers["z=H"]["ACE"] == pytest.approx(-0.001)
    ax = _strata_axes(fig)[0]
    note_bb = _note(ax, lang, e["ACE"]).get_window_extent(renderer)
    axes_bb = ax.get_window_extent(renderer)
    zero_y = ax.transData.transform((0.0, 0.0))[1]
    labels = _value_labels(ax)
    assert len(labels) == 2
    for bar, label in _pair(ax, labels):
        bb = label.get_window_extent(renderer)
        lo, hi = _bar_ys(bar, renderer)
        assert not _overlap(bb, note_bb)
        assert _inside(bb, axes_bb)
        assert bb.y0 >= lo - 1.0
        if hi - lo < bb.height + 8:       # bar shorter than the label
            assert bb.y0 >= zero_y - 1.0
        else:
            assert bb.y1 <= zero_y + 1.0


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_positive_strata_labels_stay_above_the_bar(render_effects, lang):
    fig, renderer, res = render_effects(POS, lang)
    e = res["因果効果"]["x → y=p | do(z)"]
    assert e["ACE"] > 0
    ax = _strata_axes(fig)[0]
    note_bb = _note(ax, lang, e["ACE"]).get_window_extent(renderer)
    axes_bb = ax.get_window_extent(renderer)
    labels = _value_labels(ax)
    assert len(labels) == 2
    for bar, label in _pair(ax, labels):
        assert bar.get_height() > 0
        bb = label.get_window_extent(renderer)
        _lo, hi = _bar_ys(bar, renderer)
        assert not _overlap(bb, note_bb)
        assert _inside(bb, axes_bb)
        assert bb.y0 >= hi - 1.0          # still above the bar, as before
