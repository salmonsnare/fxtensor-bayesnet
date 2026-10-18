Command-line reference
======================

.. code-block:: text

   fxtensor-bayesnet [--config PATH] [options]

``python -m fxtensor_bayesnet`` accepts the same arguments.
``--help`` prints the usage text in Japanese. ``--version`` prints the package version.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Meaning
   * - ``--config PATH``
     - Config file. Default: ``config.json`` in the current directory.
   * - ``--check``
     - Validate only. Every config error is printed. Exit code 2 on error, 0 when the file is valid. Does not write results. ``--plot`` is ignored.
   * - ``--no-write``
     - Do not write ``results.json``, ``results.md``, or ``model.mmd``. Still prints to stdout. Figures are still written when ``--plot`` is set. A snapshot is still written when ``--snapshot`` is set.
   * - ``--md``
     - Print Markdown instead of JSON.
   * - ``--lang ja|en``
     - Language of Markdown and of figure titles and legends. Default ``ja``. Does not change keys in ``results.json``.
   * - ``--json-keys ja|en``
     - Language of keys in ``results.json`` and in JSON on stdout. Default ``ja``. Snapshots in ``history/`` always use Japanese keys, so ``--diff`` keeps working.
   * - ``--snapshot``
     - Save the config and the results (Japanese keys) to ``<history-dir>/<as_of>/``. If that folder exists and ``--force`` is not set, the process exits with code 1 **before** writing results, figures, or the snapshot.
   * - ``--force``
     - Allow ``--snapshot`` to overwrite.
   * - ``--diff [PATH]``
     - Diff against a snapshot, then exit. Does not write ``results.json``. See below.
   * - ``--format-config``
     - Rewrite the config in place, one kernel row per line, then continue with the rest of the command.
   * - ``--out-dir DIR``
     - Where results and figures go. Default: the config file's directory.
   * - ``--history-dir DIR``
     - Where snapshots go. Default: ``history/`` next to the config.
   * - ``--plot``
     - Write figures to ``--out-dir``. Ignored with ``--check``. With ``--diff``, writes ``diff.png`` when a previous ``results.json`` exists.
   * - ``--plot-format F``
     - ``png`` and/or ``svg``. Several values are allowed. Default ``png``.
   * - ``--plot-dpi N``
     - PNG resolution. Default 150.

Exit codes
----------

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Code
     - When
   * - 0
     - Success. Also ``--diff`` when no earlier snapshot exists (a message is printed; a snapshot dated the same day as ``as_of`` is not chosen automatically).
   * - 1
     - ``--snapshot`` and the destination folder already exists, without ``--force``. Nothing is written.
   * - 2
     - The config is missing, is not JSON, or fails validation (including duplicate effect keys). Also a ``--diff`` path that does not exist, a folder without ``config.json``, or a comparison file that is not valid JSON.
   * - 3
     - ``--plot`` was set (and ``--check`` was not) but matplotlib is not installed. The message tells you to ``pip install 'fxtensor-bayesnet[plot]'``.

.. doctest::

   >>> import json, subprocess, sys, tempfile
   >>> from pathlib import Path
   >>> from fxtensor_bayesnet import __version__
   >>> d = Path(tempfile.mkdtemp())
   >>> cfg = {
   ...     "as_of": "2026-10-18",
   ...     "variables": {
   ...         "A": {"name": "Cloud", "labels": ["Clear", "Cloudy"]},
   ...         "B": {"name": "Rain", "labels": ["None", "Some"]},
   ...     },
   ...     "priors": {"A": {"p": [0.5, 0.5]}},
   ...     "kernels": {"B": {"parents": ["A"], "cpt": {"Clear": [0.8, 0.2], "Cloudy": [0.4, 0.6]}}},
   ... }
   >>> p = d / "config.json"
   >>> _ = p.write_text(json.dumps(cfg), encoding="utf-8")
   >>> ok = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--config", str(p), "--check"],
   ...                     capture_output=True, text=True)
   >>> ok.returncode
   0
   >>> "設定OK" in ok.stdout
   True
   >>> cfg["effects"] = [
   ...     {"treatment": "A", "outcome": "B", "name": "dup"},
   ...     {"treatment": "A", "outcome": "B", "name": "dup"},
   ... ]
   >>> _ = p.write_text(json.dumps(cfg), encoding="utf-8")
   >>> bad = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--config", str(p), "--check"],
   ...                      capture_output=True, text=True)
   >>> bad.returncode
   2
   >>> "重なる" in bad.stderr
   True
   >>> missing = subprocess.run(
   ...     [sys.executable, "-m", "fxtensor_bayesnet", "--config", str(d / "missing.json"), "--check"],
   ...     capture_output=True, text=True)
   >>> missing.returncode
   2
   >>> ver = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--version"],
   ...                      capture_output=True, text=True)
   >>> __version__ in (ver.stdout + ver.stderr)
   True

``--diff``
----------

``PATH`` may be a snapshot directory (its ``config.json`` and ``results.json`` are used)
or a ``config.json`` file. A missing path exits with code 2.

Without ``PATH``, the latest directory under ``--history-dir`` whose name matches
``YYYY-MM-DD``, contains ``config.json``, and is **strictly before** ``as_of`` is used.
A snapshot with the same date as ``as_of`` is never picked automatically.
Pass it explicitly, for example ``--diff history/2026-10-18``.

The diff lists config changes: added and removed variables, label and display-name edits,
prior changes, prior ``reason`` changes, kernel parent changes (with every old and new row),
added and removed CPT rows, kernel ``reason`` changes, and changes to scenarios, observations,
interventions, effects, sensitivity, and outputs.
When the previous ``results.json`` is present, the tool reruns inference and lists result
changes of 0.1 points or more: scenario marginals, intervention marginals, and ACEs.
It does not diff observation posteriors or sensitivity rows.
Config and result messages are Japanese.

``--diff`` returns before the normal write. Combine it with ``--plot`` to write ``diff.png``.
If there is no previous ``results.json``, a message is printed and that figure is skipped.

Output files
------------

On a normal run (not ``--check`` or ``--diff``), unless ``--no-write``:

- ``results.json`` — priors, marginals per scenario, Bayesian updates, interventions, effects, sensitivity, and checks. Key language follows ``--json-keys``.
- ``results.md`` — tables and a summary. Heading language follows ``--lang``.
- ``model.mmd`` — mermaid ``graph TD``. Roots, intermediate variables, and outputs are coloured. Labels come from the config, not from ``--lang``.

Figures
-------

``--plot`` needs matplotlib. SVGs store text as paths, so they do not depend on the viewer's fonts.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - File
     - Contents
     - When
   * - ``network.png``
     - Layered network. Roots blue, intermediate grey, outputs orange. No Graphviz.
     - Normal run
   * - ``marginals.png``
     - Bar charts of each output marginal, scenarios side by side.
     - Normal run
   * - ``updates.png``
     - Prior vs posterior of the focus variables, with total variation. Impossible observations are omitted.
     - When at least one possible observation was updated
   * - ``effects.png``
     - Observed P(Y \| X=x) next to P(Y \| do(X=x)), annotated with the observational difference and the ACE. A level with probability 0 is marked undefined instead of drawn as 0. Stratum ACEs get their own panels. If the result key is not the plain ``X → Y=y``, the panel title includes that key.
     - When ``effects`` is set
   * - ``sensitivity.png``
     - Tornado chart.
     - When ``sensitivity`` is set
   * - ``diff.png``
     - Change in marginals, in points, since the previous snapshot.
     - ``--diff --plot`` with a previous ``results.json``

:doc:`quickstart` shows these files for the weather, maintenance, and ecology examples.

A font that can display Japanese is chosen from this list, in order:
Noto Sans CJK JP, Noto Sans JP, Source Han Sans JP, IPAexGothic, IPAGothic,
Hiragino Sans, Hiragino Kaku Gothic ProN, Yu Gothic, Meiryo, MS Gothic,
TakaoGothic, VL Gothic, Noto Sans CJK SC, Noto Sans Mono CJK JP, Noto Serif CJK JP.
If none is installed, a warning is issued and the default font is used.
Set ``FXTENSOR_BAYESNET_FONT`` to a font name to override the search.

Daily workflow
--------------

.. code-block:: bash

   cd my-model
   fxtensor-bayesnet --check
   fxtensor-bayesnet --diff
   fxtensor-bayesnet --snapshot --md
   fxtensor-bayesnet --plot --no-write
