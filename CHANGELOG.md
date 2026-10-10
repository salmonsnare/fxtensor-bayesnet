# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

[0.9.0]: https://github.com/salmonsnare/fxtensor-bayesnet/releases/tag/v0.9.0
