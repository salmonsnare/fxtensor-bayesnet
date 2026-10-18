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
# fxtensor-salmon publishes its inventory under /en/ (the site root has no objects.inv).
extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "fxtensor-salmon": ("https://salmonsnare.github.io/fxtensor-salmon/en/", None),
}
intersphinx_timeout = 8

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_static"]
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
