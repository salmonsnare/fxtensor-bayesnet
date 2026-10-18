設定ファイル
============

設定は JSON オブジェクト 1 つです。``load_config`` が読み、
:func:`fxtensor_bayesnet.config.validate` が検査します。
``--check`` は見つかった問題をすべて表示し、終了コード 2 で終わります。推論は始まりません。

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - キー
     - 必須
     - 内容
   * - ``as_of``
     - はい
     - 基準日。慣例は ``YYYY-MM-DD``。スナップショットのフォルダ名はこの文字列です。
       ``--check`` は日付として解釈しません。
       引数なしの ``--diff`` は、名前が ``YYYY-MM-DD`` に一致し、文字列として ``as_of`` より小さいフォルダだけを見ます。
   * - ``horizon``
     - 
     - 結果に出ます（``ホライズン`` / ``horizon``）。省略すると JSON の ``null``。
   * - ``note``
     - 
     - 結果の注意書き（``注意`` / ``note``）。省略すると JSON の ``null``。
   * - ``variables``
     - はい
     - 変数 id → ``{"name": 表示名, "labels": [ラベル, ...]}``。
       ``name`` は任意で、無いときは Markdown が id を使います。
       ラベルは 2 個以上、重複なし。このオブジェクト内の順は自由で、根を先に置く必要はありません。
   * - ``priors``
     - はい
     - 根の id → ``{"p": [ラベルごとの確率], "reason": 任意の文章}``。
       ``p`` はラベルと同じ長さ、負の値なし、和は 1（許容差 ``1e-6``）。
   * - ``kernels``
     - はい
     - 子の id → ``{"parents": [親の id], "cpt": {...}, "reason": 任意の文章}``。
   * - ``scenarios``
     - 
     - シナリオ名 → ``{根の id: 差し替える事前分布}``。
       基本シナリオは常に追加され、ここには書きません。
       根でない変数はエラーで、メッセージは ``interventions`` を示します。
   * - ``observations``
     - 
     - 観測のリスト。下記。
   * - ``interventions``
     - 
     - 介入名 → ``{変数: ラベルまたは分布}``。下記。
   * - ``effects``
     - 
     - 因果効果の問いのリスト。下記。
   * - ``sensitivity``
     - 
     - オブジェクト 1 つ、またはそのリスト。下記。
   * - ``outputs``
     - 
     - 結果の表に出す変数 id。省略、または空リストは、根以外のすべて。
       ``outputs`` に入れない非根は中間変数です（図では灰色）。

ここに無いトップレベルのキーは検証されません。ファイルには残るので
:func:`fxtensor_bayesnet.config.config_hash` は変わりますが、推論は読みません。
``reason`` は事前分布と核に書け、変わると ``--diff`` が報告します。``results.json`` にはコピーされません。

各変数は ``priors`` か ``kernels`` のどちらか一方にだけ置きます。
親の関係に循環はできません。
変数は最大 52 個です。超えると :class:`fxtensor_bayesnet.network.Net` が ``ValueError`` を出します。
``--check`` はネットワークを組まないので、この上限は見ません。

条件付き確率表
--------------

CPT のキーは、親のラベルを ``parents`` の順に ``|`` でつないだものです。
``"parents": ["load", "age"]`` なら ``"High|Old"`` のようになります。
値は子のラベル順の確率です。各行の和は 1（許容差 ``1e-6``）。
親ラベルのすべての組合せに行が必要です。
``--check`` は足りない行と余分な行を別々に報告します。

観測
----

2 つの書き方があります。1 項目の中で混ぜることはできません。

1 変数へのハードな証拠：

.. code-block:: json

   {"var": "rain", "label": "Some", "focus": ["cloud"]}

``focus`` は、事前から事後への動き（ポイント差、全変動距離、ナット単位の KL）と、
全変動距離による順位を出す変数です。省略すると根のすべてです。未知の id はエラーです。

同時の証拠と、ソフト証拠：

.. code-block:: json

   {"name": "Sensor suggests Some",
    "evidence": {"rain": {"likelihood": [0.2, 0.9]}},
    "focus": ["cloud"]}

- 文字列の値はハードな証拠（そのラベル）です。
- ``{"likelihood": [...]}`` はソフト証拠です。ラベルごとの非負の重み
  ``P(報告 | 真のラベル)`` で、和が 1 である必要はありません。少なくとも 1 つは正です。
  長さはラベル数と一致します。
- ``evidence`` の項目はすべて同時に条件付けます。
- ``name`` があれば結果のキーになります。無ければ証拠から作ります。
  例は ``rain=Some を観測``、ソフト証拠なら ``rain~L[0.2, 0.9] を観測``。
- ``--json-keys en`` では、末尾が `` を観測`` のキーからその接尾辞を除きます。
  自分で書いた ``name`` は訳しません。

モデルの下で確率 0 の観測は、実行を止めません。
``run`` は警告し、その更新だけを省き、``不可能な観測`` / ``impossible`` が true、
確率 0、空の事後として記録します。ほかの観測は通常どおり計算します。
``updates.png`` はその観測を描きません。
:meth:`Engine.condition_raw <fxtensor_bayesnet.inference.Engine.condition_raw>` と
:meth:`Engine.posterior_raw <fxtensor_bayesnet.inference.Engine.posterior_raw>` を直接呼ぶと、
この場合は今までどおり ``ValueError`` です。

証拠がすべてハードなとき、更新は ``観測前の確率`` / ``evidence_probability``
（証拠の事前確率）を持ちます。
1 つでもソフトだと、代わりに ``尤度の期待値`` / ``expected_likelihood`` です。
尤度の重みの期待値 Σ P(x) L(x) で、確率ではありません。1 を超えることがあり、
Markdown は百分率ではなく数値で出します。

介入
----

.. code-block:: json

   "interventions": {
     "Force cloudy": {"cloud": "Cloudy"},
     "Half cloudy": {"cloud": [0.5, 0.5]}
   }

ラベルはハードな do() です。リストは確率的な介入で、分布として妥当である必要があります
（長さ、非負、和が 1）。1 つのオブジェクトに複数の変数を書くと、同時に介入します。
介入した変数に入る辺を切り、下流を計算し直します。上流は事前分布のままです。
条件付けとの違いはここです。

根の事前分布を差し替えるには、これまでどおり ``scenarios`` を使います。
根への do() は、同じ分布を置くシナリオと同じです。
結果には、介入の下での ``outputs`` の周辺分布が出ます。

因果効果
--------

各項目は、処置の各水準について P(結果 | do(処置)) と P(結果 | 処置)、
ACE、観測上の差、その差（交絡）を求めます。

.. code-block:: json

   {"treatment": "cloud", "outcome": "rain", "outcome_label": "Some",
    "treated": "Cloudy", "control": "Clear"}

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - フィールド
     - 内容
   * - ``treatment``
     - 変数 X。必須。``outcome`` と別の変数。
   * - ``outcome``
     - 変数 Y。必須。
   * - ``outcome_label``
     - Y のラベル y。既定は Y の最後のラベル。
   * - ``treated``
     - 処置群にする X の水準。既定は X の最後のラベル。
   * - ``control``
     - 対照群にする X の水準。既定は X の最初のラベル。
       既定を埋めたあと、``treated`` と ``control`` は別のラベルである必要があります。
   * - ``given_do``
     - 任意の ``{Z: ラベル}``。Z も介入で固定します。層別 ACE。
   * - ``given``
     - 任意の ``{W: ラベル}``。介入したモデルの中で W を観測します。条件付き ACE。
   * - ``name``
     - 任意の、空でない文字列。結果のキーになり、``名前`` / ``name`` としても残ります。

``given_do`` と ``given`` の値は、ラベルのリスト、ラベル 1 つ、または ``null``（全ラベル）です。
変数が複数あると、ラベルの直積が層になります。層の名前は ``Z=z`` または ``Z=z, W=w`` です。
Z は処置でも結果でもなく、同じ変数を両方のキーには書けません。
未知の変数、未知のラベル、ラベルの重複は設定エラーです。

``given_do`` の層 z ごとに、各水準 x の P(Y=y | do(X=x), do(Z=z)) と、
ACE(z) = P(y | do(treated), do(z)) − P(y | do(control), do(z)) が出ます。
``given`` では X を介入で固定し、そのモデルで Z を条件付けます。
P(Y=y | do(X=x), Z=z) です。do(X=x) の下で確率 0 の層は未定義
（JSON は ``null``、Markdown は ``—``）で、``run`` は警告します。
最後の層の ACE から最初の層の ACE を引いた値は
``層間のACEの差(最後−最初)`` / ``ace_difference_last_minus_first`` です
（層が 1 つ、またはどちらかの ACE が未定義なら ``null``）。

結果のキー
~~~~~~~~~~

:func:`fxtensor_bayesnet.causal.effect_key` がキーを作ります。

- ``name`` があればそれ。
- 無ければ ``X → Y=y``。
- ``given_do`` だけ：``X → Y=y | do(Z)``。変数が複数なら、オブジェクトの順で ``X → Y=y | do(Z, W)``。
- ``given`` だけ：``X → Y=y | W``。
- 両方：``X → Y=y | do(Z), W``。

``given_do`` / ``given`` の中で選んだラベルはキーに入りません。入るのは変数 id です。
層の指定が無い効果は、これらのフィールドが無かった頃と同じく、平文の ``X → Y=y`` のままです。

.. doctest::

   >>> from fxtensor_bayesnet.causal import effect_key
   >>> V = {"X": {"labels": ["a", "b"]}, "Y": {"labels": ["no", "yes"]},
   ...      "Z": {"labels": ["0", "1"]}, "W": {"labels": ["0", "1"]}}
   >>> effect_key(V, {"treatment": "X", "outcome": "Y"})
   'X → Y=yes'
   >>> effect_key(V, {"treatment": "X", "outcome": "Y", "outcome_label": "no",
   ...                "given_do": {"Z": ["0", "1"]}})
   'X → Y=no | do(Z)'
   >>> effect_key(V, {"treatment": "X", "outcome": "Y", "given": {"W": None}})
   'X → Y=yes | W'
   >>> effect_key(V, {"treatment": "X", "outcome": "Y", "given_do": {"Z": None},
   ...                "given": {"W": None}})
   'X → Y=yes | do(Z), W'
   >>> effect_key(V, {"treatment": "X", "outcome": "Y", "name": "by Z"})
   'by Z'

キーの重複
~~~~~~~~~~

同じキーになる項目が 2 つあると設定エラーです。典型は、同じ ``given_do`` の変数で
ラベルのリストだけが違う場合で、キーはどちらも ``X → Y=y | do(Z)`` です。
``--check`` はそれを報告し、終了コード 2 で終わります。
:func:`fxtensor_bayesnet.config.validate` と :func:`fxtensor_bayesnet.run` は
最後の 1 件だけを残さずに :class:`~fxtensor_bayesnet.ConfigError` を送出します。
``name`` を変えるか、層にする変数を変えてください。
例外の文は日本語で、``重なる`` を含みます。

.. doctest::

   >>> from fxtensor_bayesnet import ConfigError, run
   >>> from fxtensor_bayesnet.config import validate
   >>> cfg = {
   ...     "as_of": "2026-10-18",
   ...     "variables": {
   ...         "X": {"labels": ["a", "b"]},
   ...         "Y": {"labels": ["no", "yes"]},
   ...     },
   ...     "priors": {"X": {"p": [0.5, 0.5]}},
   ...     "kernels": {"Y": {"parents": ["X"], "cpt": {"a": [0.8, 0.2], "b": [0.3, 0.7]}}},
   ...     "effects": [
   ...         {"treatment": "X", "outcome": "Y", "name": "same"},
   ...         {"treatment": "X", "outcome": "Y", "outcome_label": "yes", "name": "same"},
   ...     ],
   ... }
   >>> try:
   ...     validate(cfg)
   ... except ConfigError as exc:
   ...     print("重なる" in str(exc))
   True
   >>> try:
   ...     run(cfg)
   ... except ConfigError as exc:
   ...     print("重なる" in str(exc))
   True

処置の水準の確率が 0 のとき、P(結果 | 処置 = その水準) は未定義です。
JSON の値は ``null`` で、その水準は ``観測確率0の水準`` / ``zero_probability_levels`` に入り、
``run`` は警告します。do() 側の確率と ACE は定義されたままです。
Markdown は ``—`` です。``effects.png`` はその水準の観測の棒を描かず、
"undefined (P=0)" と記します。

感度分析
--------

``sensitivity`` はオブジェクト 1 つか、そのリストです。

.. code-block:: json

   {"target": "rain", "label": "Some", "delta": 0.05, "top": 10}

``label`` の既定は ``target`` の最後のラベルです。
``delta`` の既定は 0.05 で、0 より大きく 1 より小さい必要があります。
``top`` の既定は 10 で、1 以上の整数です。

対象の祖先集合にある各パラメータ（事前分布の 1 要素、または CPT の 1 行の 1 要素）を
±delta 動かし、[0, 1] に切り詰めます。同じ行の残りは、和が 1 になるよう比例的に縮めます。
ラベルが 2 つの行は、最後のラベルだけ動かします。
結果は幅（P(target = label) の変化の絶対値、ポイント）の大きい順です。
保存するのは上位 ``top`` 件です。パラメータ数は、その切り詰め前の個数です。
``--plot`` はトルネード図を描きます。

``--check`` が拒否するもの
--------------------------

``validate`` は問題をすべて集めて :class:`~fxtensor_bayesnet.ConfigError` を送出します。
メッセージは ``設定エラー:`` で始まり、日本語です。
``--check`` はそれを表示して終了コード 2 です。正しいファイルは ``設定OK`` を出して終了コード 0 で、
結果は書きません。

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 状況
     - 結果
   * - ``as_of`` が無い
     - ConfigError
   * - ``variables`` が空
     - ConfigError
   * - ラベルが 2 個未満、または重複
     - ConfigError
   * - 変数が ``priors`` と ``kernels`` の両方にある、またはどちらにも無い
     - ConfigError
   * - 事前分布か核が、``variables`` に無い変数を指している
     - ConfigError
   * - 分布の長さが違う、負の値がある、または和が 1 でない（許容差 ``1e-6``）
     - ConfigError
   * - 親が未定義
     - ConfigError
   * - CPT の行が足りない、または余分
     - ConfigError
   * - シナリオが根でない変数を差し替える
     - ConfigError（メッセージは ``interventions`` に言及する）
   * - 観測がオブジェクトでない、ラベルが未知、または ``evidence`` と ``var`` / ``label`` が同時にある
     - ConfigError
   * - 尤度の長さが違う、負の重みがある、または正の重みが無い
     - ConfigError
   * - ``focus`` が未知の変数
     - ConfigError
   * - 介入が空、未知の変数、またはラベルでも分布でもない
     - ConfigError
   * - 効果に ``treatment`` / ``outcome`` が無い、両方が同じ変数、未知のラベル、または ``treated`` と ``control`` が同じ
     - ConfigError
   * - ``given_do`` / ``given`` の形が違う、処置か結果を指す、ラベルが重複、または同じ変数が両方にある
     - ConfigError
   * - ``name`` があるが、空でない文字列ではない
     - ConfigError
   * - 2 つの効果が同じ結果キーになる
     - ``validate`` と ``run`` の ConfigError。``--check`` は終了コード 2
   * - ``sensitivity`` の対象やラベルが未知、``delta`` が (0, 1) の外、または ``top`` が 1 以上の整数でない
     - ConfigError
   * - ``outputs`` が未知の変数
     - ConfigError
   * - グラフに循環がある
     - ConfigError
   * - 変数が 52 個を超える
     - ネットワークを組むときの ``ValueError``。``--check`` では出ない
   * - ファイルが無い、または JSON でない
     - CLI の終了コード 2（``ConfigError`` ではない）

実際に行った検算が直接計算と合わないとき、``run`` は ``AssertionError`` を出して止まります。
許容差は、全同時・周辺・介入が ``1e-12``、条件付けと ``bayesian_inversion`` が ``1e-9`` です。
