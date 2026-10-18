"""English text for autodoc when ``language`` is ``en``.

The source docstrings are Japanese. This extension replaces them only while the
English docs are built, so both API pages are complete and the library text
itself stays unchanged. A public docstring that is still Japanese fails the
build (``sphinx-build -W``).
"""
from __future__ import annotations

import re

from sphinx.util import logging

logger = logging.getLogger(__name__)

_JP = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

EN = {
    "fxtensor_bayesnet": (
        "Config-driven inference for discrete Bayesian networks, built on fxtensor-salmon."
    ),
    "fxtensor_bayesnet.causal": (
        "Interventions (the do-operator, by graph surgery), the average causal effect (ACE), "
        "and sensitivity of conditional probability tables."
    ),
    "fxtensor_bayesnet.causal.do_vector": (
        "Turn an intervention spec into a distribution vector. A label string is a hard do(); "
        "a list of probabilities is a stochastic intervention."
    ),
    "fxtensor_bayesnet.causal.do_config": (
        "Graph surgery. For each variable in ``do`` (label or distribution), drop the incoming "
        "kernel and place that distribution in the priors. The original config is not modified.\n"
        "\n"
        "The derived config has empty ``scenarios``, ``observations``, ``interventions``, "
        "``effects``, and ``sensitivity``."
    ),
    "fxtensor_bayesnet.causal.given_levels": (
        "Normalize one ``given_do`` / ``given`` value to a list of labels. "
        "``None`` means every label; a string is one label; a list is used as given."
    ),
    "fxtensor_bayesnet.causal.strata": (
        "Build strata from ``{variable: [labels, ...]}``.\n"
        "\n"
        "Returns a list of ``(stratum name, {variable: label})``. "
        "The name is ``Z=z``, or ``Z=z, W=w`` when several variables are set."
    ),
    "fxtensor_bayesnet.causal.effect_spec": (
        "Fill defaults for one ``effects`` item: ``outcome_label`` is the last label of the "
        "outcome, ``treated`` the last label of the treatment, ``control`` the first.\n"
        "\n"
        "``given_do`` and ``given`` are normalized to ``{variable: [labels, ...]}``. "
        "Missing specifications become empty dicts."
    ),
    "fxtensor_bayesnet.causal.effect_key": (
        "Result key for one ``effects`` item. ``name`` is used when it is set. "
        "Otherwise the key is ``X → Y=y``, plus the stratification when there is one: "
        "``X → Y=y | do(Z)``, ``X → Y=y | W``, or ``X → Y=y | do(Z), W``."
    ),
    "fxtensor_bayesnet.causal.effect_key_errors": (
        "Error messages for ``effects`` items whose result keys collide. Empty when they do not."
    ),
    "fxtensor_bayesnet.causal.sensitivity_specs": (
        "Normalize ``sensitivity`` (one object or a list) to a list and fill defaults."
    ),
    "fxtensor_bayesnet.causal.perturb_row": (
        "Set entry ``j`` of row ``p`` to ``new_pj`` and rescale the other entries proportionally "
        "so the row still sums to 1."
    ),
    "fxtensor_bayesnet.causal.sensitivity": (
        "One-at-a-time sensitivity of ``P(target=label)`` to each parameter "
        "(each root-prior entry, and each entry of each CPT row in the target's ancestor set).\n"
        "\n"
        "Each parameter is moved by ±delta and clipped to [0, 1]; the rest of the same row "
        "moves proportionally. A two-label row moves only its last label.\n"
        "\n"
        "Returns ``(baseline probability, number of parameters, top rows sorted by swing)``."
    ),
    "fxtensor_bayesnet.cli": (
        "Command-line interface for fxtensor-bayesnet.\n"
        "\n"
        "Usage::\n"
        "\n"
        "    fxtensor-bayesnet [--config PATH] [options]\n"
        "\n"
        "``--config PATH``\n"
        "    Config file (default: ``config.json`` in the current directory).\n"
        "``--out-dir DIR``\n"
        "    Where ``results.json``, ``results.md``, and ``model.mmd`` are written "
        "(default: the config file's directory).\n"
        "``--history-dir DIR``\n"
        "    Where snapshots are written (default: ``history/`` next to the config).\n"
        "``--no-write``\n"
        "    Do not write those files; print to stdout only.\n"
        "``--md``\n"
        "    Print Markdown instead of JSON.\n"
        "``--check``\n"
        "    Validate the config only (exit code 2 on error).\n"
        "``--snapshot``\n"
        "    Save the config and results under ``<history-dir>/<as_of>/``. "
        "An existing folder needs ``--force``.\n"
        "``--force``\n"
        "    Allow ``--snapshot`` to overwrite.\n"
        "``--diff [PATH]``\n"
        "    Diff the config (and results) against a previous snapshot. "
        "Without PATH, the latest snapshot dated strictly before ``as_of``.\n"
        "``--format-config``\n"
        "    Rewrite the config so each kernel row is one line.\n"
        "``--plot``\n"
        "    Write figures to the output directory (``network``, ``marginals``, ``updates``, "
        "``effects``, ``sensitivity``; with ``--diff``, also ``diff``). "
        "Requires matplotlib: ``pip install 'fxtensor-bayesnet[plot]'``.\n"
        "``--plot-format``\n"
        "    One or more of ``png`` and ``svg`` (default ``png``).\n"
        "``--plot-dpi N``\n"
        "    PNG resolution (default 150).\n"
        "``--lang ja|en``\n"
        "    Language of Markdown and figure text (default ``ja``). "
        "Does not change keys in ``results.json``.\n"
        "``--json-keys ja|en``\n"
        "    Language of keys in ``results.json`` and stdout JSON (default ``ja``). "
        "Snapshots always keep Japanese keys.\n"
        "\n"
        "The text above is the English form of the module docstring. "
        "``--help`` prints the Japanese text stored on the module."
    ),
    "fxtensor_bayesnet.cli.build_parser": "Return the command-line ArgumentParser.",
    "fxtensor_bayesnet.cli.main": (
        "Run the CLI and return a status code: 0 success, 1 snapshot already exists, "
        "2 config or input error, 3 plotting dependency missing."
    ),
    "fxtensor_bayesnet.config": "Loading, validation, formatting, and hashing of configuration files.",
    "fxtensor_bayesnet.config.ConfigError": "Raised when a configuration file is invalid.",
    "fxtensor_bayesnet.config.load_config": "Load a JSON config, validate it, and return it.",
    "fxtensor_bayesnet.config.topo_order": (
        "Topological order of the variables (parents first). Raises ConfigError on a cycle."
    ),
    "fxtensor_bayesnet.config.validate": (
        "Check a config. Every problem is reported together in one ConfigError."
    ),
    "fxtensor_bayesnet.config.config_hash": (
        "Short hash of the config contents. Key order does not matter."
    ),
    "fxtensor_bayesnet.config.dump_config": (
        "Readable JSON with each kernel row and each label list on one line, "
        "so a diff is readable line by line."
    ),
    "fxtensor_bayesnet.history": "Daily snapshots and diffs.",
    "fxtensor_bayesnet.history.zero": "Round and normalize -0.0 to 0.0.",
    "fxtensor_bayesnet.history.snapshot_dir": "Snapshot directory, ``history_dir/as_of``.",
    "fxtensor_bayesnet.history.check_snapshot_free": (
        "Check that the snapshot directory is free. Call this before writing anything."
    ),
    "fxtensor_bayesnet.history.resolve_diff_path": (
        "Resolve a ``--diff`` argument to ``(config.json path, results.json path or None)``.\n"
        "\n"
        "A directory uses the ``config.json`` and ``results.json`` inside it. "
        "Raises FileNotFoundError when the config cannot be found."
    ),
    "fxtensor_bayesnet.history.snapshot": (
        "Write the config and the results (Japanese keys) to ``history_dir/<as_of>/``.\n"
        "\n"
        "Raises FileExistsError when the directory exists and ``force`` is false. "
        "Returns the directory written."
    ),
    "fxtensor_bayesnet.history.previous_snapshot": (
        "Path to ``config.json`` of the latest dated snapshot in ``history_dir`` strictly "
        "before ``as_of``, or None."
    ),
    "fxtensor_bayesnet.history.diff_configs": (
        "Lines describing changes between two configs. "
        "A single 'no config changes' line when there are none."
    ),
    "fxtensor_bayesnet.history.diff_results": (
        "Lines for result changes of at least ``threshold_pt`` points. "
        "A single 'no result changes' line when there are none."
    ),
    "fxtensor_bayesnet.i18n": (
        "Display strings for Markdown and figures. Keys in ``results.json`` do not follow "
        "``--lang`` (English keys are ``--json-keys en``)."
    ),
    "fxtensor_bayesnet.i18n.t": (
        "Return ``TEXT[lang][key]``. Formats it when keyword arguments are given. "
        "Raises ValueError for an unsupported language."
    ),
    "fxtensor_bayesnet.i18n.scenario_name": (
        "Map the internal scenario name for the base scenario to the display language."
    ),
    "fxtensor_bayesnet.i18n.obs_text": (
        "Observation heading (``kind`` is ``obs_header`` or ``obs_annot``). "
        "Hard evidence uses the prior probability of the evidence (percent). "
        "An entry that contains soft evidence uses the expected likelihood."
    ),
    "fxtensor_bayesnet.i18n.obs_name": (
        "Strip the `` を観測`` suffix from an observation result key, leaving ``var=label``."
    ),
    "fxtensor_bayesnet.inference": (
        "Inference: marginals per scenario, Bayesian updates, root-cause inversion, "
        "interventions, causal effects, sensitivity, and einsum cross-checks."
    ),
    "fxtensor_bayesnet.inference.as_dict": (
        "Map labels to probabilities rounded to ``nd`` decimal places (-0.0 becomes 0.0)."
    ),
    "fxtensor_bayesnet.inference.movement": (
        "Movement from prior to posterior: difference, total variation, and KL divergence."
    ),
    "fxtensor_bayesnet.inference.evidence_vectors": (
        "Turn evidence ``{variable: label | {\"likelihood\": [...]} | [...]}`` into "
        "``{variable: likelihood vector}``."
    ),
    "fxtensor_bayesnet.inference.normalize_observation": (
        "Turn one ``observations`` item into ``(result key, evidence dict, legacy form?)``."
    ),
    "fxtensor_bayesnet.inference.Engine": (
        "Marginals, conditioning, and root-cause inversion for one set of priors.\n"
        "\n"
        "``mode`` is ``joint`` when the joint has at most ``max_joint`` elements, "
        "and ``ancestral`` above that."
    ),
    "fxtensor_bayesnet.inference.Engine.__init__": (
        "Build an engine for ``net`` and root priors ``{variable: list of probabilities}``."
    ),
    "fxtensor_bayesnet.inference.Engine.marginal": (
        "Marginal distribution. In ``joint`` mode, discard axes with the library wiring kernel "
        "(or sum the joint directly when that kernel would exceed the size limit). "
        "In ``ancestral`` mode, einsum over the ancestor set (variable elimination)."
    ),
    "fxtensor_bayesnet.inference.Engine.pair": (
        "Joint state I → a ⊗ b as an FXTensor. When the wiring kernel would exceed the size "
        "limit, sum the joint directly (``joint`` mode) or use einsum (``ancestral`` mode)."
    ),
    "fxtensor_bayesnet.inference.Engine.condition_raw": (
        "Unrounded posterior of each variable given ``obs=label``. "
        "Uses ``conditionalization(2)`` on the pair ``(obs, v)``. "
        "Raises ValueError when that observation has probability 0."
    ),
    "fxtensor_bayesnet.inference.Engine.condition": (
        "``condition_raw`` rounded to four decimal places, as dicts."
    ),
    "fxtensor_bayesnet.inference.Engine.posterior_raw": (
        "Condition on several pieces of evidence at once (hard evidence and soft / likelihood "
        "evidence), by variable elimination (einsum over the ancestor set).\n"
        "\n"
        "``evidence`` maps a variable to a label or to ``{\"likelihood\": [...]}``. "
        "Returns ``(probability of the evidence, or expected likelihood, {variable: posterior})``. "
        "When ``targets`` is omitted, every variable that is not hard evidence is returned. "
        "Raises ValueError when the evidence has probability 0."
    ),
    "fxtensor_bayesnet.inference.Engine.inversion_roots": (
        "``inversion_roots_raw`` rounded to four decimal places, as dicts."
    ),
    "fxtensor_bayesnet.inference.Engine.inversion_size": (
        "Number of elements in the dense tensor built for root-cause inversion "
        "(all roots × ``obs``)."
    ),
    "fxtensor_bayesnet.inference.Engine.inversion_roots_raw": (
        "Unrounded root posteriors. Builds the kernel (all roots) → ``obs`` with einsum and "
        "inverts it with ``FXTensor.bayesian_inversion``."
    ),
    "fxtensor_bayesnet.inference.run": (
        "Run a config and return the result dict. Raises AssertionError when a cross-check "
        "that was actually performed does not match."
    ),
    "fxtensor_bayesnet.keys": (
        "English keys for ``results.json`` (``--json-keys en`` / :func:`to_english_keys`).\n"
        "\n"
        "The internal result dict and history snapshots stay in Japanese, which is the "
        "canonical form. English keys are an output-only translation. Names chosen by the "
        "user (variables, labels, scenarios, and so on) are not translated. "
        "The mapping is positional, so a label that happens to be a Japanese key is safe."
    ),
    "fxtensor_bayesnet.keys.to_english_keys": (
        "Return a new dict with English keys for a ``run()`` result. "
        "User-chosen names and values are unchanged. "
        "An observation key ``var=label を観測`` becomes ``var=label``."
    ),
    "fxtensor_bayesnet.keys.KEY_MAP": (
        "Japanese key to English key, for the fixed keys only. "
        "The same pairs are listed in the output-format reference."
    ),
    "fxtensor_bayesnet.network": (
        "The network: states and Markov kernels built with FXTensor, and direct einsum computation."
    ),
    "fxtensor_bayesnet.network.Net": (
        "A Bayesian network built from a config. "
        "Each child variable's conditional probability table is a Markov kernel."
    ),
    "fxtensor_bayesnet.network.Net.__init__": (
        "Build variables, labels, and kernels from a config. "
        "Raises ValueError when there are more than 52 variables."
    ),
    "fxtensor_bayesnet.network.Net.ancestors": (
        "Ancestor set of ``names``, including the names themselves. "
        "Variables in ``cut`` are treated as if their edges from parents were removed "
        "(graph surgery for do)."
    ),
    "fxtensor_bayesnet.network.Net.joint_size": (
        "Number of elements in the full joint (the product of the label counts)."
    ),
    "fxtensor_bayesnet.network.Net.contract": (
        "Variable elimination: einsum only the factors in the ancestor set, and return the "
        "unnormalized joint of ``targets``.\n"
        "\n"
        "``evidence`` maps a variable to a likelihood vector (hard evidence is one-hot). "
        "The result is ``P(targets, evidence)``.\n"
        "``do`` maps a variable to a distribution that replaces that variable's kernel "
        "(graph surgery, truncated factorization).\n"
        "``cpt`` maps a variable to an ndarray that replaces its kernel, or a root prior "
        "(used by sensitivity analysis).\n"
        "``free_roots=True`` returns the kernel from the roots to ``targets`` "
        "(root priors are not multiplied in).\n"
        "The full joint is not built; ``np.einsum(..., optimize=True)`` keeps intermediate "
        "tensors small."
    ),
    "fxtensor_bayesnet.network.Net.einsum_query": (
        "Direct einsum for the joint of ``targets``, using only the ancestor set. "
        "With ``free_roots=True``, the kernel from the roots to ``targets``."
    ),
    "fxtensor_bayesnet.network.Net.brute_force_joint": (
        "Full joint of every variable in one einsum, for cross-checks. Axis order is ``self.order``."
    ),
    "fxtensor_bayesnet.network.Net.prior_state": (
        "Root priors as an FXTensor state (tensor product). "
        "``priors`` maps each root to a list of probabilities."
    ),
    "fxtensor_bayesnet.network.Net.build_joint": (
        "Build the joint state: tensor-product the roots, then add each child in topological "
        "order with ``graph_extend``.\n"
        "\n"
        "Axis order is roots (``self.roots`` order) then children (topological order). "
        "That can differ from ``self.order``."
    ),
    "fxtensor_bayesnet.network.wiring_size": (
        "Number of elements in the dense tensor that ``wiring(in_labels, out_idx)`` would build."
    ),
    "fxtensor_bayesnet.network.wiring": (
        "Deterministic wiring kernel X1⊗…⊗Xn → X at ``out_idx`` "
        "(the labelled form of copy / swap / discard)."
    ),
    "fxtensor_bayesnet.network.graph_extend": (
        "Extend a state I→X by a kernel k:(parents)→Y, producing I→X⊗Y. "
        "Does not build a dense identity."
    ),
    "fxtensor_bayesnet.network.self_test_graph_extend": (
        "Check that ``graph_extend`` matches the same extension written only with the library."
    ),
    "fxtensor_bayesnet.plot": (
        "Figures. matplotlib is optional: ``pip install 'fxtensor-bayesnet[plot]'``.\n"
        "\n"
        "- ``plot_network`` — network structure (layered layout; roots, intermediate, and outputs)\n"
        "- ``plot_marginals`` — marginals of each output, scenarios side by side\n"
        "- ``plot_updates`` — prior vs posterior of the focus variables, per observation\n"
        "- ``plot_diff`` — change in marginals (points) since a previous snapshot\n"
        "- ``plot_effects`` — conditioning vs intervention, when ``effects`` is set\n"
        "- ``plot_sensitivity`` — tornado chart, when ``sensitivity`` is set"
    ),
    "fxtensor_bayesnet.plot.PlotError": "Raised when a figure cannot be drawn (for example, matplotlib is not installed).",
    "fxtensor_bayesnet.plot.setup_font": (
        "Pick a font that can display Japanese and set it. "
        "Warns and keeps the default font when none is found."
    ),
    "fxtensor_bayesnet.plot.node_kinds": (
        "Map each variable to ``root``, ``middle``, or ``output``. "
        "A non-root that is not in ``outputs`` is ``middle``."
    ),
    "fxtensor_bayesnet.plot.layered_layout": (
        "Layered layout (a small Sugiyama-style pass).\n"
        "\n"
        "A layer is the longest-path distance from the roots. An edge that skips two or more "
        "layers gets a dummy node on each layer in between. Order within a layer is improved "
        "with a few barycenter sweeps.\n"
        "\n"
        "Returns ``layers`` (elements per layer; a dummy is ``(\"__d\", parent, child, layer)``), "
        "``level``, ``parents``, and ``chains`` (``(parent, child)`` → dummies along that edge)."
    ),
    "fxtensor_bayesnet.plot.plot_network": "Write the network diagram to ``path`` and return that path.",
    "fxtensor_bayesnet.plot.plot_marginals": (
        "Write marginals of the output variables (scenarios side by side) to ``path`` and return that path."
    ),
    "fxtensor_bayesnet.plot.plot_updates": (
        "Write prior vs posterior of the focus variables for each observation to ``path``. "
        "Returns None when there is nothing to draw."
    ),
    "fxtensor_bayesnet.plot.plot_diff": (
        "Write the change in marginals, in points, from the previous result to the new one. "
        "Returns None when there is no scenario to compare."
    ),
    "fxtensor_bayesnet.plot.diff_deltas": (
        "``{scenario: [(variable, label, change in points)]}``, "
        "only for scenarios and variables present in both results."
    ),
    "fxtensor_bayesnet.plot.plot_effects": (
        "Write causal-effect panels (observation vs intervention, and stratum ACEs) to ``path``. "
        "Returns None when there are no effects."
    ),
    "fxtensor_bayesnet.plot.plot_sensitivity": (
        "Write the sensitivity tornado chart to ``path``. Returns None when there is no sensitivity block."
    ),
    "fxtensor_bayesnet.plot.plot_all": (
        "Write ``network``, ``marginals``, and ``updates`` (and ``effects`` / ``sensitivity`` when "
        "present) under ``out_dir``. Returns the paths that were written."
    ),
    "fxtensor_bayesnet.report": "Markdown result tables and a mermaid structure diagram.",
    "fxtensor_bayesnet.report.render_markdown": (
        "Return the results as Markdown. ``lang`` is ``ja`` or ``en`` "
        "(heading language; the result dict is still read with Japanese keys)."
    ),
    "fxtensor_bayesnet.report.render_mermaid": (
        "Mermaid ``graph TD`` of the network. Roots, intermediate variables, and outputs are coloured."
    ),
    "fxtensor_bayesnet.report.safe": (
        "Mermaid node id. Characters other than letters, digits, and ``_`` are replaced with a short hash."
    ),
}


# The CLI module docstring is the Japanese ``--help`` text. It is not
# reStructuredText, so the Japanese API build substitutes this summary.
JA = {
    "fxtensor_bayesnet.cli": (
        "コマンドライン。\n"
        "\n"
        "使い方は ``fxtensor-bayesnet [--config PATH] [options]`` 。\n"
        "``--help`` は、このモジュールに書いてある日本語の説明をそのまま出す"
        "（その文は reStructuredText ではない）。\n"
        "各フラグ、終了コード、図、 ``--diff`` の規則は CLI リファレンスに書いた。"
    ),
    "fxtensor_bayesnet.report.render_markdown": (
        "結果の Markdown を返す。lang は ``ja`` または ``en`` "
        "(見出しの言語。結果辞書は日本語キーのまま読む)。"
    ),
}


def _process(app, what, name, obj, options, lines):
    if app.config.language == "ja":
        text = JA.get(name)
        if text is not None:
            lines[:] = text.splitlines()
        return
    if app.config.language != "en":
        return
    text = EN.get(name)
    if text is None:
        if _JP.search("\n".join(lines)):
            logger.warning(
                "no English docstring for %s",
                name,
                type="docstring",
                subtype="untranslated",
            )
        return
    lines[:] = text.splitlines()


def setup(app):
    app.connect("autodoc-process-docstring", _process)
    return {"version": "1", "parallel_read_safe": True}
