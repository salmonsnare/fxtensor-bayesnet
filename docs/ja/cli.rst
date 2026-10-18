コマンドライン
==============

.. code-block:: text

   fxtensor-bayesnet [--config PATH] [options]

``python -m fxtensor_bayesnet`` も同じ引数です。
``--help`` は使い方を日本語で出します。``--version`` はパッケージのバージョンを出します。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - オプション
     - 内容
   * - ``--config PATH``
     - 設定ファイル。既定はカレントディレクトリの ``config.json``。
   * - ``--check``
     - 検証だけ。設定エラーはすべて表示。エラーなら終了コード 2、正しければ 0。結果は書かない。``--plot`` は無視。
   * - ``--no-write``
     - ``results.json``、``results.md``、``model.mmd`` を書かない。標準出力への表示は行う。``--plot`` の図は書く。``--snapshot`` のスナップショットも書く。
   * - ``--md``
     - JSON の代わりに Markdown を出す。
   * - ``--lang ja|en``
     - Markdown と図の見出し・凡例の言語。既定は ``ja``。``results.json`` のキーは変えない。
   * - ``--json-keys ja|en``
     - ``results.json`` と標準出力の JSON のキーの言語。既定は ``ja``。``history/`` のスナップショットは常に日本語キーなので、``--diff`` は版をまたいで動く。
   * - ``--snapshot``
     - 設定と結果（日本語キー）を ``<history-dir>/<as_of>/`` に保存。そのフォルダが既にあり ``--force`` が無いと、結果も図もスナップショットも書く前に終了コード 1。
   * - ``--force``
     - ``--snapshot`` の上書きを許す。
   * - ``--diff [PATH]``
     - スナップショットとの差分を出して終了。``results.json`` は書かない。下記。
   * - ``--format-config``
     - 設定をその場で、核の 1 行が 1 行になるよう書き戻してから、残りの処理を続ける。
   * - ``--out-dir DIR``
     - 結果と図の出力先。既定は設定ファイルのフォルダ。
   * - ``--history-dir DIR``
     - スナップショットの場所。既定は設定ファイルの隣の ``history/``。
   * - ``--plot``
     - 図を ``--out-dir`` に書く。``--check`` のときは無視。``--diff`` と一緒で、前回の ``results.json`` があれば ``diff.png`` を書く。
   * - ``--plot-format F``
     - ``png`` と ``svg`` の一方または両方。複数指定できる。既定は ``png``。
   * - ``--plot-dpi N``
     - PNG の解像度。既定は 150。

終了コード
----------

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - コード
     - いつ
   * - 0
     - 成功。以前のスナップショットが無い ``--diff`` もここ（メッセージを出して終わる。``as_of`` と同じ日付のスナップショットは自動では選ばない）。
   * - 1
     - ``--snapshot`` で出力先が既にあり、``--force`` が無い。何も書かない。
   * - 2
     - 設定が無い、JSON でない、または検証に失敗（効果のキーの重複を含む）。``--diff`` のパスが無い、``config.json`` の無いフォルダ、比較ファイルが JSON でない、もここ。
   * - 3
     - ``--plot`` があり（``--check`` ではない）matplotlib が無い。メッセージは ``pip install 'fxtensor-bayesnet[plot]'`` を示す。

.. doctest::

   >>> import json, subprocess, sys, tempfile
   >>> from pathlib import Path
   >>> from fxtensor_bayesnet import __version__
   >>> d = Path(tempfile.mkdtemp())
   >>> cfg = {
   ...     "as_of": "2026-10-18",
   ...     "variables": {
   ...         "A": {"name": "Cloud", "labels": ["Clear", "Cloudy"]},
   ...         "B": {"name": "Rain", "labels": ["None", "Some"]},
   ...     },
   ...     "priors": {"A": {"p": [0.5, 0.5]}},
   ...     "kernels": {"B": {"parents": ["A"], "cpt": {"Clear": [0.8, 0.2], "Cloudy": [0.4, 0.6]}}},
   ... }
   >>> p = d / "config.json"
   >>> _ = p.write_text(json.dumps(cfg), encoding="utf-8")
   >>> ok = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--config", str(p), "--check"],
   ...                     capture_output=True, text=True)
   >>> ok.returncode
   0
   >>> "設定OK" in ok.stdout
   True
   >>> cfg["effects"] = [
   ...     {"treatment": "A", "outcome": "B", "name": "dup"},
   ...     {"treatment": "A", "outcome": "B", "name": "dup"},
   ... ]
   >>> _ = p.write_text(json.dumps(cfg), encoding="utf-8")
   >>> bad = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--config", str(p), "--check"],
   ...                      capture_output=True, text=True)
   >>> bad.returncode
   2
   >>> "重なる" in bad.stderr
   True
   >>> missing = subprocess.run(
   ...     [sys.executable, "-m", "fxtensor_bayesnet", "--config", str(d / "missing.json"), "--check"],
   ...     capture_output=True, text=True)
   >>> missing.returncode
   2
   >>> ver = subprocess.run([sys.executable, "-m", "fxtensor_bayesnet", "--version"],
   ...                      capture_output=True, text=True)
   >>> __version__ in (ver.stdout + ver.stderr)
   True

``--diff``
----------

``PATH`` はスナップショットのディレクトリ（中の ``config.json`` と ``results.json`` を使う）か、
``config.json`` ファイルです。無いパスは終了コード 2 です。

``PATH`` を省略すると、``--history-dir`` の下で名前が ``YYYY-MM-DD`` に一致し、
``config.json`` があり、``as_of`` より前で最新のディレクトリを使います。
``as_of`` と同じ日付のスナップショットは、自動では選びません。
明示するときは ``--diff history/2026-10-18`` のように渡します。

差分に出る設定の変更は、変数の追加と削除、ラベルと表示名、事前分布、事前分布の ``reason``、
核の親の変更（古い行と新しい行のすべて）、CPT の行の追加と削除、核の ``reason``、
シナリオ、観測、介入、効果、感度、``outputs`` です。
前回の ``results.json`` があれば推論をやり直し、0.1 ポイント以上の変化を出します。
対象はシナリオの周辺分布、介入の周辺分布、ACE です。
観測の事後や感度の行は比べません。
設定と結果のメッセージは日本語です。

``--diff`` は通常の書き込みの前に戻ります。``--plot`` と組むと ``diff.png`` を書きます。
前回の ``results.json`` が無ければメッセージを出して、その図は省きます。

出力ファイル
------------

通常の実行（``--check`` でも ``--diff`` でもない）で、``--no-write`` でなければ：

- ``results.json`` — 事前分布、シナリオごとの周辺、ベイズ更新、介入、効果、感度、検算。キーの言語は ``--json-keys``。
- ``results.md`` — 表と要約。見出しの言語は ``--lang``。
- ``model.mmd`` — mermaid の ``graph TD``。根、中間、出力を色分け。ラベルは設定から取り、``--lang`` では変わりません。

図
--

``--plot`` には matplotlib が必要です。SVG は文字をパスにするので、見る側のフォントに依存しません。

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - ファイル
     - 内容
     - 出るとき
   * - ``network.png``
     - 階層レイアウトのネットワーク。根は青、中間は灰、出力は橙。Graphviz は不要。
     - 通常の実行
   * - ``marginals.png``
     - 各出力の周辺分布の棒グラフ。シナリオを並べる。
     - 通常の実行
   * - ``updates.png``
     - 注目変数の事前と事後。全変動距離付き。不可能な観測は省く。
     - 更新できた観測が 1 件以上あるとき
   * - ``effects.png``
     - 観測の P(Y \| X=x) と P(Y \| do(X=x)) を並べ、観測上の差と ACE を添える。確率 0 の水準は 0 の棒ではなく未定義。層別 ACE は別パネル（層ごとに棒。負の値は棒の内側、棒が短いときは 0 のすぐ上。全体の ACE は破線で、そのパネルの凡例に出す）。結果のキーが平文の ``X → Y=y`` でなければ、パネルの題にそのキーを出す。
     - ``effects`` があるとき
   * - ``sensitivity.png``
     - トルネード図。
     - ``sensitivity`` があるとき
   * - ``diff.png``
     - 前回のスナップショットからの周辺分布の変化（ポイント）。
     - ``--diff --plot`` で、前回の ``results.json`` があるとき

:doc:`quickstart` に、天気・設備保全・生態学の例で出したこれらのファイルがあります。

日本語を表示できるフォントを、次の順で探します。
Noto Sans CJK JP、Noto Sans JP、Source Han Sans JP、IPAexGothic、IPAGothic、
Hiragino Sans、Hiragino Kaku Gothic ProN、Yu Gothic、Meiryo、MS Gothic、
TakaoGothic、VL Gothic、Noto Sans CJK SC、Noto Sans Mono CJK JP、Noto Serif CJK JP。
どれも無ければ警告し、既定のフォントを使います。
探す順を上書きするには、環境変数 ``FXTENSOR_BAYESNET_FONT`` にフォント名を置きます。

日次の流れ
----------

.. code-block:: bash

   cd my-model
   fxtensor-bayesnet --check
   fxtensor-bayesnet --diff
   fxtensor-bayesnet --snapshot --md
   fxtensor-bayesnet --plot --no-write
