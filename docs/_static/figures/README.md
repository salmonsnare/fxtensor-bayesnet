# TikZ figures

When an issue is resolved, add a corresponding TikZ figure to the documentation.

- Put the TikZ source (`.tex`) in this directory, next to the rendered SVG or PNG.
- Include that figure on the relevant page in both `docs/en` and `docs/ja`, with a caption in that language.
- The Sphinx build treats warnings as errors (`sphinx -W`).

Plot screenshots generated from the examples live in `docs/_static/images/` (`en/` and `ja/`). Those are separate from the TikZ figures in this directory.

# TikZ の図

Issue を解決したときは、対応する TikZ の図をドキュメントに追加します。

- TikZ のソース（`.tex`）と、描画した SVG または PNG を、このディレクトリに置きます。
- `docs/en` と `docs/ja` の該当ページの両方に、その言語のキャプション付きで載せます。
- Sphinx のビルドは警告をエラーにします（`sphinx -W`）。

例題から生成したプロットの画像は `docs/_static/images/`（`en/` と `ja/`）にあります。このディレクトリの TikZ の図とは別です。
