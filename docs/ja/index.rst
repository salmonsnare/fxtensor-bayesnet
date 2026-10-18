fxtensor-bayesnet
==================

これは日本語のドキュメントです。
英語版はサイトのルートにあります: `English <../index.html>`_。

fxtensor-bayesnet は、設定ファイル駆動の離散ベイジアンネットワーク推論ツールです。
`fxtensor-salmon <https://github.com/salmonsnare/fxtensor-salmon>`_
（:doc:`ドキュメント <fxtensor-salmon:index>`）の上に組まれています。
仮定と確率は JSON ファイルに書くので、コードを変えずに見直せます。
実行すると、シナリオの比較、観測による信念の更新、do 演算による因果の問いに答えます。

計算の仕方：

- 各変数は離散で、有限個のラベル（例：``晴れ`` / ``曇り``）をとります。
- 親を持たない変数（根）には事前分布を書きます。子には条件付き確率表を書きます。
  親のラベルの組合せごとに、子の分布（マルコフ核）が 1 つあります。
- 同時分布は、fxtensor-salmon の :class:`fxtensor_salmon.FXTensor` で状態と核を合成して作ります。
  そこからシナリオごとの周辺分布、観測による条件付け、``bayesian_inversion`` による根の事後を計算します。
- 毎回、ライブラリの結果を NumPy の ``einsum`` と丸め前の値で突き合わせます。ずれると実行は止まります。
  :doc:`output` を参照してください。
- 同時分布の要素数が 200 万を超えると、祖先集合だけで計算します。値は同じです。
  条件付けでは 100 万要素を超える密な配線核を作らず、同時分布の直接和か変数消去を使います。
- ``interventions`` は、根かどうかにかかわらず任意の変数へ do 演算（グラフ手術）をかけます。
  ``effects`` は平均因果効果（ACE）を、観測上の差と並べて出します。
- 1 件の観測に、複数の証拠とソフト証拠（尤度）を書けます。
  ``sensitivity`` は、ある確率がどの事前分布・CPT の値に強く依存するかを順位付けします。
- ``--plot`` はネットワーク、シナリオ比較、ベイズ更新、前回からの変化を PNG または SVG で書きます。
  matplotlib が必要です。

.. note::

   設定に書く確率は利用者の仮定です。
   結果はその仮定の帰結です。
   このツールは仮定そのものを検証しません。

PyPI のパッケージ名は ``fxtensor-bayesnet`` です。このドキュメントはバージョン |release| に対応します。
ライセンスは MIT License です。

.. note::

   Issue を解決したときは、対応する TikZ の図をこのドキュメントに追加します。
   TikZ のソースと描画した図は ``docs/_static/figures/`` に置きます。
   詳細は ``docs/_static/figures/README.md`` です。
   英語版と日本語版の両方に載せます。

.. toctree::
   :maxdepth: 1
   :caption: 目次

   install
   quickstart
   config
   cli
   output
   変更履歴 <changelog>
   api
