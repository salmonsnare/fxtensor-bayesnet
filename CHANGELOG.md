# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

[0.9.1]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.9.1
[0.9.0]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.9.0
