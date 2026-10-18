クイックスタート
==================

次のモデルは、変数が 2 つの天気のスケッチです。ラベルと確率は説明用です。
``cloud`` は根です。``rain`` は ``cloud`` だけに依存します。

- P(rain = Some | Clear) = 0.1
- P(rain = Some | Cloudy) = 0.6
- P(cloud = Clear) = 0.6 なので、P(rain = Some) = 0.6 × 0.1 + 0.4 × 0.6 = 0.3
- シナリオ ``A: Mostly cloudy`` は雲の事前分布を ``[0.2, 0.8]`` に差し替えるので、
  P(rain = Some) = 0.2 × 0.1 + 0.8 × 0.6 = 0.5
- rain = Some を観測したあと、P(cloud = Cloudy | rain = Some) = (0.6 × 0.4) / 0.3 = 0.8

ラベルは利用者が自由に付けられます。この例は英語のラベルを使っていますが、
``晴れ`` / ``曇り`` のような日本語のラベルでも同じ計算になります。

.. doctest::

   >>> import json
   >>> from fxtensor_bayesnet import run, to_english_keys
   >>> from fxtensor_bayesnet.config import validate
   >>> cfg = json.loads(r"""
   ... {
   ...   "as_of": "2026-10-18",
   ...   "horizon": "Next day",
   ...   "note": "Illustrative probabilities, not measurements",
   ...   "variables": {
   ...     "cloud": {"name": "Cloud cover", "labels": ["Clear", "Cloudy"]},
   ...     "rain": {"name": "Rain", "labels": ["None", "Some"]}
   ...   },
   ...   "priors": {"cloud": {"p": [0.6, 0.4]}},
   ...   "kernels": {
   ...     "rain": {
   ...       "parents": ["cloud"],
   ...       "cpt": {"Clear": [0.9, 0.1], "Cloudy": [0.4, 0.6]}
   ...     }
   ...   },
   ...   "scenarios": {"A: Mostly cloudy": {"cloud": [0.2, 0.8]}},
   ...   "observations": [{"var": "rain", "label": "Some", "focus": ["cloud"]}],
   ...   "outputs": ["rain"]
   ... }
   ... """)
   >>> validate(cfg)
   >>> en = to_english_keys(run(cfg))
   >>> en["scenarios"]["base"]["marginals"]["rain"]["Some"]
   0.3
   >>> en["scenarios"]["A: Mostly cloudy"]["marginals"]["rain"]["Some"]
   0.5
   >>> round(en["updates_base"]["rain=Some"]["evidence_probability"], 4)
   0.3
   >>> en["updates_base"]["rain=Some"]["posterior"]["cloud"]["Cloudy"]
   0.8
   >>> en["checks"]["mode"]
   'joint'
   >>> en["checks"]["max_err_joint_vs_einsum"] < 1e-9
   True

``run`` は日本語キーで返します。:func:`fxtensor_bayesnet.to_english_keys` は固定のキーだけを訳し、
付けた名前（``cloud``、``Clear``、``A: Mostly cloudy``）はそのまま残します。
基本シナリオのキー ``基本`` は ``base`` になります。

コマンドライン
--------------

同じ JSON を ``config.json`` として保存し、検証します。``--check`` は結果を書きません。

.. code-block:: bash

   fxtensor-bayesnet --check
   fxtensor-bayesnet --no-write --md --lang en
   fxtensor-bayesnet --no-write --json-keys en

正しいファイルに対する ``--check`` の終了コードは 0 です。
ファイルが無い、JSON でない、設定エラー、のいずれも終了コード 2 です。

リポジトリの大きめの例
----------------------

``examples/`` は PyPI の配布物には入りません。
`リポジトリ <https://github.com/salmonsnare/fxtensor-bayesnet>`_ から取得してください。

- ``examples/weather/`` — 気圧、湿度、雲、降水、屋外イベント、交通。
  日本語の ``config.json`` と英語の ``config.en.json`` は数値が同じです。
- ``examples/maintenance/`` — 負荷、経過年数、保守、摩耗、振動、軸受温度、故障。同じく両言語です。
- ``examples/ecology_host_parasitoid/`` — 英語のみ。介入、因果効果、同時・ソフト証拠、感度分析。
  数値はすべて説明用の仮の値です。

これらの図は ``python docs/make_images.py``（``plot`` 追加依存が必要）で再生成し、
``docs/images/`` と ``docs/images/en/`` に置きます。
