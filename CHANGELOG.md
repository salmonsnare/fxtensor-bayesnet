# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.11.3] - 2026-10-18

### Fixed
- `effects.png` stratum-ACE panels (`given_do` / `given`): labels on negative bars were drawn below the bar, so they
  overlapped the overall-ACE note in the lower right and the overall-ACE dashed line cut through the digits. A negative
  label now sits inside the bar, near the end toward 0 (just above 0 when the bar is too short to hold it). The overall
  ACE is the legend entry of that dashed line, placed above the axes. Value labels keep a white background where the
  line passes behind them (#33).

## [0.11.2] - 2026-10-15

### Fixed
- Checks reported a max error of `0.0` for comparisons that were never made. In `ancestral` mode the joint/soft
  evidence cross-check (`einsum(条件付け)との最大差`) and the full-joint check (`einsum(全同時)との最大差`) were skipped
  but kept their initial `0.0`. A max-error entry is now `null` unless the comparison was actually made (also
  without observations), observations whose conditioning check was skipped are listed under
  `条件付けの検算を省略した観測` (`conditioning_check_skipped`), and Markdown states which checks were not run.
  Posteriors and all values in `joint` mode are unchanged (#26).

## [0.11.1] - 2026-10-15

### Fixed
- Two `effects` items with the same (treatment, outcome, outcome_label) were stored under the same key `X → Y=y`, so
  the first one (with its `given_do` / `given` results) was silently overwritten and `--check` passed. The key now
  includes the stratification when there is one (`X → Y=y | do(Z)`, `X → Y=y | W`, `X → Y=y | do(Z), W`); effects
  without `given_do` / `given` keep `X → Y=y`. Keys that still collide are a config error in `--check`, and `run()`
  raises `ConfigError` instead of overwriting (#24).

### Added
- `effects[].name`: optional result key for an effect (also stored as `名前` / `name`). `effects.png` shows the key in
  the panel title when it differs from `X → Y=y`, so panels of the same effect can be told apart (#24).

### Changed
- Result keys of effects that use `given_do` / `given` (introduced in 0.11.0) gain the `| …` suffix described above.

## [0.11.0] - 2026-10-12

### Added
- `effects[].given_do`: stratified ACE. For each stratum z of one or more other variables (also fixed by intervention),
  P(outcome | do(X=x), do(Z=z)) for every level and ACE(z), plus the difference between the last and the first
  stratum (effect modification). Stored under `層別ACE(given_do)` (`stratified_ace_given_do`) (#22).
- `effects[].given`: conditional ACE. X is fixed by intervention and Z is observed in the intervened model,
  P(outcome | do(X=x), Z=z); a stratum with probability 0 under do(X=x) is `null` with a warning. Stored under
  `条件付きACE(given)` (`conditional_ace_given`) and cross-checked against the graph-surgery network (#22).
- Values of `given_do` / `given` may be a list of labels, one label or `null` (all labels); several variables give
  the combinations of their labels. `--check` reports unknown variables or labels, the treatment or outcome used as
  a given variable and a variable listed in both keys. Markdown gets a table per stratification and `effects.png` a
  panel of stratum ACEs. Effects without these keys give exactly the same output as before (#22).

### Fixed
- `effects.png` drew the observed bar of a treatment level with probability 0 as a bar of height 0 labelled "0",
  although P(outcome | X=x) is undefined there (`null` in `results.json`). No bar is drawn now; the level is marked
  "undefined (P=0)" with a note in the panel. `run()` issues a `UserWarning` and the effect lists such levels in
  `観測確率0の水準` (`zero_probability_levels`); Markdown adds a note next to the `—` (#21).

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

[0.11.3]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.11.3
[0.11.2]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.11.2
[0.11.1]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.11.1
[0.11.0]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.11.0
[0.10.1]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.10.1
[0.10.0]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.10.0
[0.9.1]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.9.1
[0.9.0]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.9.0
