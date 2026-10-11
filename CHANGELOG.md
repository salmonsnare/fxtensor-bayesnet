# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.10.1] - 2026-10-12

### Fixed
- Soft evidence: the value reported for an observation that contains soft (likelihood) evidence is the expected
  likelihood Σ P(x) L(x), which can exceed 1, but it was stored as `観測前の確率` (`evidence_probability`) and shown as a
  percentage (e.g. `620.0%` for weights `[2, 9]` on a 0.4 / 0.6 prior). Such entries now use the key `尤度の期待値`
  (`expected_likelihood`), and Markdown and `updates.png` show it as "expected likelihood 6.2" instead of a
  probability. Hard-evidence entries and all posteriors are unchanged (#17).
- An observation with probability 0 under the model raised `ValueError` and aborted the whole run (a Traceback and
  exit code 1 from the CLI). The run now continues: a `UserWarning` is issued, that update is skipped and recorded
  with `不可能な観測: true` (`impossible`), probability 0 and an empty posterior, and the Markdown output says that the
  observation has probability 0 (`updates.png` leaves it out, as there is nothing to compare). `Engine.condition_raw`
  and `Engine.posterior_raw` still raise `ValueError` when called directly (#18).
- `updates.png`: a long observation name squashed every panel into thin bars, because the row heading counted as part
  of the first panel when the layout was computed. The heading is now left out of the layout, wrapped to the figure
  width and placed above the panel titles, so the panels keep their width (#19).

## [0.10.0] - 2026-10-11

### Added
- `interventions`: do() on any variable, root or non-root, by graph surgery. A label gives a hard intervention and a
  probability list a stochastic one. Results (`介入`) are cross-checked against a direct truncated-factorisation
  einsum (`介入:切断因子分解との最大差`) (#3).
- `effects`: for each `{treatment, outcome}`, P(outcome | do(X=x)) and P(outcome | X=x) for every level, plus the ACE,
  the observational difference and the gap between them (confounding) (`因果効果`) (#3).
- `observations[].evidence`: several observations conditioned on jointly, and soft (likelihood / virtual) evidence
  `{"likelihood": [...]}`, computed by variable elimination. Optional `name` for the result key. The `var`/`label`
  form still works and gives identical results (#4).
- `sensitivity`: one-at-a-time sensitivity of P(target=label) to every prior and CPT entry in the target's ancestor
  set (±delta with proportional co-variation), sorted by swing (`感度分析`) (#5).
- `--plot` also writes `effects.png` (observed vs intervened) and `sensitivity.png` (tornado chart) when the
  corresponding keys are set (#7). Markdown output has sections for interventions, effects and sensitivity.
- `--json-keys ja|en` and `to_english_keys(res)` write `results.json` / stdout JSON with English keys. The default
  stays Japanese, and snapshots always keep the Japanese keys so `--diff` works across versions. Names chosen by the
  user are never translated (#6).
- Python API: `Engine.posterior_raw(evidence)`, `Net.contract(...)` (variable elimination with evidence, do() and CPT
  overrides), `do_config(cfg, do)`, `fxtensor_bayesnet.causal.sensitivity`.
- `--diff` reports changes to `interventions`, `effects` and `sensitivity`, and changes in intervention marginals
  and ACEs.
- Example `examples/ecology_host_parasitoid/` (English): observation vs intervention in a host–parasite–parasitoid
  cascade, with a script that checks the ACEs against an independent truncated factorisation. All numbers are
  illustrative placeholders (#8).

### Fixed
- The built-in cross-check `einsum(全同時)との最大差` failed (`AssertionError`) when a root variable was declared in
  `variables` after a non-root variable. The declaration order no longer matters (#1).
- `MemoryError: 配線核が大きすぎます` when observations were applied with a joint of roughly 560k–2M elements
  (e.g. 13 three-state variables). In joint mode, marginals and pairs now fall back to summing the joint directly
  when the dense wiring kernel would exceed `MAX_WIRING` (lowered from 5M to 1M elements). The 13-variable chain now
  peaks at about 50 MB. The `bayesian_inversion` cross-check is skipped (and listed in the checks) when the
  roots × observation tensor would be too large (#2).

### Changed
- `scenarios` on a non-root variable now suggests `interventions` in the validation error.

## [0.9.1] - 2026-10-10

### Fixed
- `--diff <folder>` crashed with `IsADirectoryError`. A snapshot folder is now accepted (its `config.json` and
  `results.json` are used); a missing path or a folder without `config.json` gives a clear error and exit code 2.
- `--snapshot` checked for an existing snapshot only after writing results and figures. The check now happens first;
  without `--force` nothing is written and the exit code is 1.
- `--diff` did not report changes to the `reason` of root priors, and omitted the rows of kernels whose parents
  changed. It now reports prior reasons, the parent change with all old and new rows, and added/removed rows.
- The `einsum(条件付け)との最大差` and `bayesian_inversionとconditionalizationの最大差` checks compared values rounded
  to 4 decimals. They now compare unrounded values (and the tolerance is tightened accordingly).
- `-0.0` could appear in diffs and Markdown output; values are now normalized to `0.0`.

### Documentation
- READMEs (English and Japanese) state that a snapshot with the same date as `as_of` is never picked automatically by
  `--diff`, and document folder paths for `--diff` and the new `--snapshot` behaviour.

## [0.9.0] - 2026-10-10

First public release.

### Added
- Config-driven discrete Bayesian network inference built on fxtensor-salmon (`FXTensor`):
  marginals per scenario, Bayesian updating on observations, and root-cause inference via `bayesian_inversion`.
- Automatic cross-check of every result against a direct numpy `einsum` computation; ancestor-set
  computation for large joint distributions.
- Config validation (`--check`) reporting all errors at once (row sums, missing/extra rows, undefined parents, cycles, …).
- Daily snapshots (`--snapshot`, `--force`) and diffs against the previous snapshot (`--diff`).
- Output as JSON, Markdown (`--md`) and a colour-coded mermaid diagram (`model.mmd`); `--no-write`, `--out-dir`,
  `--history-dir`, `--format-config`.
- Figures with matplotlib (optional extra `[plot]`): network structure with a layered layout, marginals by scenario,
  prior vs posterior for observations, and day-over-day changes (`--plot`, `--plot-format png|svg`, `--plot-dpi`).
  Japanese fonts are detected automatically.
- `--lang ja|en` for Markdown and figure text.
- Examples: weather and equipment maintenance, each in Japanese and English.
- README in English and Japanese.

[0.10.1]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.10.1
[0.10.0]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.10.0
[0.9.1]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.9.1
[0.9.0]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.9.0
