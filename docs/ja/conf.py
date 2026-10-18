import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared_conf import *  # noqa: E402,F401,F403

language = "ja"
html_title = "fxtensor-bayesnet"
html_theme_options = {**html_theme_options, "source_directory": "docs/ja/"}
