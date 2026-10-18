"""Settings shared by the English and Japanese Sphinx builds."""
from __future__ import annotations

import os
import sys

_DOCS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DOCS, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_DOCS, "_ext"))

from fxtensor_bayesnet import __version__  # noqa: E402

project = "fxtensor-bayesnet"
author = "Tadachika Oki"
copyright = "2026, Tadachika Oki"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "myst_parser",
    "en_docstrings",
]
# The fxtensor-salmon documentation site did not publish objects.inv when these
# pages were written, so links to it are plain URLs. If that inventory appears,
# intersphinx is enabled for future cross-references; the prose does not depend on it.
_SALMON_DOCS = "https://salmonsnare.github.io/fxtensor-salmon/"


def _salmon_inventory():
    import urllib.request

    try:
        req = urllib.request.Request(_SALMON_DOCS + "objects.inv", method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as resp:
            return 200 <= getattr(resp, "status", 0) < 300
    except Exception:
        return False


if _salmon_inventory():
    extensions.append("sphinx.ext.intersphinx")
    intersphinx_mapping = {"fxtensor-salmon": (_SALMON_DOCS, None)}
    intersphinx_timeout = 8

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# Symlinks inside each language tree, so the paths stay inside the Sphinx source directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["language.css"]

html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/salmonsnare/fxtensor-bayesnet/",
    "source_branch": "main",
    "navigation_with_keys": True,
}

autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "none"

myst_heading_anchors = 3
