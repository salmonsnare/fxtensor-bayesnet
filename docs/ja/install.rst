インストール
============

Python 3.8 以上。実行時の依存は NumPy と
`fxtensor-salmon <https://github.com/salmonsnare/fxtensor-salmon>`_（``fxtensor-salmon>=1.2.2,<2``）です。
図を描かない限り matplotlib は不要です。

.. code-block:: bash

   pip install fxtensor-bayesnet
   pip install "fxtensor-bayesnet[plot]"

コマンドは ``fxtensor-bayesnet`` です。``python -m fxtensor_bayesnet`` も同じプログラムです。

開発
----

.. code-block:: bash

   git clone https://github.com/salmonsnare/fxtensor-bayesnet.git
   cd fxtensor-bayesnet
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -e ".[dev]"
   pytest

``[dev]`` は pytest と matplotlib を入れます。通常のインストールには足しません。

このドキュメントをビルドする
----------------------------

ドキュメント用の追加依存は Python 3.10 以上が必要です（Sphinx 8）。
ライブラリ本体は Python 3.8 でも動きます。

.. code-block:: bash

   pip install -e ".[docs]"
   # または: pip install -r docs/requirements.txt && pip install -e .
   make -C docs html doctest

``make html`` は英語版を ``docs/_build/html/`` に、日本語版を ``docs/_build/html/ja/`` に書きます。
警告はエラーにします。
``make doctest`` は両言語のページに埋め込んだ例を実行します。

``docs/_build/html/index.html`` を開いてください。同じサイトは ``main`` ブランチから
https://salmonsnare.github.io/fxtensor-bayesnet/ へ公開されます。
GitHub Pages のソースは設定で **GitHub Actions** にしてください。
ワークフローは ``.github/workflows/docs.yml`` です。
