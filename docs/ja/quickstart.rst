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

下の図は README が参照しているものです。
``python docs/make_images.py`` （``plot`` 追加依存が必要）で再生成します。
再生成先は ``docs/images/`` と ``docs/images/en/`` です。
このサイト用のコピーは ``docs/_static/images/`` にあり、公開ページに含まれます。
日本語のページは日本語ラベルの図を使います。例の確率はすべて説明用です。

天気
~~~~

``examples/weather/`` は気圧、湿度、雲、降水、屋外イベント、交通です。
日本語の ``config.json`` と英語の ``config.en.json`` は数値が同じです。
シナリオは「A:台風接近」と「B:乾燥した晴天続き」です。
観測は、大雨・渋滞・中止から気圧配置と湿度を逆算します。

.. figure:: _static/images/ja/weather_network.png
   :alt: 天気の例のネットワーク構造
   :width: 100%

   天気の例：ネットワーク構造（``--plot`` の ``network.png``）。

.. figure:: _static/images/ja/weather_marginals.png
   :alt: 天気の例の周辺分布
   :width: 100%

   天気の例：周辺分布のシナリオ比較（``marginals.png``）。

.. figure:: _static/images/ja/weather_updates.png
   :alt: 天気の例のベイズ更新
   :width: 100%

   天気の例：ベイズ更新の事前と事後（``updates.png``）。
   「大雨」を観測すると、気圧配置の「低気圧接近」が 25% から 67% に上がります。

.. figure:: _static/images/ja/weather_diff.png
   :alt: 天気の例の前回からの変化量
   :width: 100%

   天気の例：前回からの変化量（``--diff --plot`` の ``diff.png``）。
   翌日に気圧配置と湿度の事前分布を見直した例です。
   A と B は気圧配置をシナリオで上書きし、B は湿度も上書きします。
   そのため変化は小さく、B は変化しません。

設備保全
~~~~~~~~

``examples/maintenance/`` は負荷、経過年数、保守、摩耗、振動、軸受温度、故障です。同じく両言語です。
シナリオは「A:繁忙期（高負荷）」と「B:保守体制の強化」です。
観測は、振動の異常、停止、温度の上昇から負荷・年数・保守を逆算します。
``wear`` （部品摩耗）は ``outputs`` に無い中間変数なので、構造図では灰色です。

.. figure:: _static/images/ja/maintenance_network.png
   :alt: 設備保全の例のネットワーク構造
   :width: 100%

   設備保全の例：ネットワーク構造（``network.png``）。

.. figure:: _static/images/ja/maintenance_marginals.png
   :alt: 設備保全の例の周辺分布
   :width: 100%

   設備保全の例：周辺分布のシナリオ比較（``marginals.png``）。

.. figure:: _static/images/ja/maintenance_updates.png
   :alt: 設備保全の例のベイズ更新
   :width: 100%

   設備保全の例：ベイズ更新（``updates.png``）。

生態学
~~~~~~

``examples/ecology_host_parasitoid/`` は英語のみです。
介入、因果効果、同時・ソフト証拠、感度分析を使います。
数値はすべて説明用の仮の値です。
図の文字も英語です。
``ecology_sensitivity.png`` はその例の README の図です。
プロジェクトの README が載せているのは因果効果の図です。

.. figure:: _static/images/en/ecology_effects.png
   :alt: 生態学の例の観測と介入
   :width: 100%

   生態学の例：観測と介入（``effects.png``）。
   数値は説明用の仮の値です。

.. figure:: _static/images/en/ecology_sensitivity.png
   :alt: 生態学の例の感度分析
   :width: 100%

   生態学の例：感度分析のトルネード図（``sensitivity.png``）。
   数値は説明用の仮の値です。
