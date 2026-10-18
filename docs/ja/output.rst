出力形式
========

``results.json`` は JSON オブジェクトです。キーは既定で日本語で、0.9 から変わっていません。
``--json-keys en`` と :func:`fxtensor_bayesnet.to_english_keys` は、同じ木を英語のキーで書きます。
自分で付けた名前（変数 id、ラベル、シナリオ名、介入名、効果の ``name``）は訳しません。
末尾が `` を観測`` の観測キーはその接尾辞を落とし、それ以外の観測キーはそのままです。
検算の備考の文は、どちらのキー言語でも日本語のままです。
スナップショットは常に日本語キーです。

結果の確率は小数 4 桁に丸めるので、分布の和がちょうど 1 にならないことがあります。
JSON の ``null`` は Python の ``None`` です。比べていない検算か、定義されない条件付き確率で、
数値の 0 ではありません。

トップレベル
------------

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - 日本語
     - 英語
     - 内容
   * - ``基準日``
     - ``as_of``
     - 設定の ``as_of``。
   * - ``ホライズン``
     - ``horizon``
     - ``horizon``。無ければ ``null``。
   * - ``注意``
     - ``note``
     - ``note``。無ければ ``null``。
   * - ``設定ハッシュ``
     - ``config_hash``
     - キーをソートした設定 JSON の SHA-256 の、先頭 12 文字（16 進）。
   * - ``事前分布(基本)``
     - ``priors_base``
     - 基本の事前分布。``{根: {ラベル: 確率}}``。
   * - ``シナリオ``
     - ``scenarios``
     - 基本を含む各シナリオ。下記。
   * - ``ベイズ更新(基本)``
     - ``updates_base``
     - 基本の事前分布の下での、観測 1 件ごと。観測が無ければ空のオブジェクト。
   * - ``介入``
     - ``interventions``
     - 設定に ``interventions`` があるときだけ。
   * - ``因果効果``
     - ``effects``
     - 設定に ``effects`` があるときだけ。
   * - ``感度分析``
     - ``sensitivity``
     - 設定に ``sensitivity`` があるときだけ。
   * - ``検算``
     - ``checks``
     - 突き合わせの報告。常にある。

シナリオ
--------

基本シナリオのキーは ``基本`` / ``base`` です。ほかのキーは設定の名前です。
各値は次のとおりです。

.. list-table::
   :header-rows: 1
   :widths: 20 22 58

   * - 日本語
     - 英語
     - 内容
   * - ``上書き``
     - ``overrides``
     - そのシナリオの差し替え事前分布（``{根: [確率]}``）。基本は ``{}``。
   * - ``周辺``
     - ``marginals``
     - ``{出力変数: {ラベル: 確率}}``。

ベイズ更新
----------

キーは、観測に ``name`` があればそれ、無ければ説明に `` を観測`` を付けたもの
（``rain=Some を観測``、ソフト証拠なら ``rain~L[0.2, 0.9] を観測``）です。

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - 日本語
     - 英語
     - 内容
   * - ``証拠``
     - ``evidence``
     - 単一のハードなラベルではないときにある。ソフト証拠は ``{"尤度": [重み]}`` / ``{"likelihood": [重み]}``。
   * - ``観測前の確率``
     - ``evidence_probability``
     - ハードな証拠の事前確率。小数 4 桁。ソフトな証拠が 1 つでもあると無い。
   * - ``尤度の期待値``
     - ``expected_likelihood``
     - ソフトな証拠があるときの Σ P(x) L(x)。確率ではなく、1 を超えうる。ハードだけなら無い。
   * - ``不可能な観測``
     - ``impossible``
     - 確率 0 で更新を省いたとき ``true``。そうでなければキー自体が無い。
   * - ``事後``
     - ``posterior``
     - ``{変数: {ラベル: 確率}}``。不可能な観測では空。ハードな証拠の変数は除く。
   * - ``注目変数の事前``
     - ``focus_prior``
     - ``focus`` の変数の事前（既定は根のすべて）。
   * - ``注目変数の動き``
     - ``focus_movement``
     - ハードな証拠ではない注目変数の動き。不可能な観測では空。
   * - ``動きの大きさ順(TV)``
     - ``ranking_by_tv``
     - その変数を、全変動距離の大きい順に。

動きの各値：

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - 日本語
     - 英語
     - 内容
   * - ``差(pt)``
     - ``delta_pt``
     - 事後ひく事前。ポイント。小数 1 桁。
   * - ``全変動距離TV``
     - ``tv_distance``
     - 全変動距離。小数 4 桁。
   * - ``KL(事後\|\|事前)[nat]``
     - ``kl_posterior_prior_nat``
     - KL(事後 \|\| 事前)。ナット。小数 4 桁。

介入
----

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - 日本語
     - 英語
     - 内容
   * - ``介入内容``
     - ``do``
     - ハードなラベルを one-hot に広げたあとの ``{変数: {ラベル: 確率}}``。
   * - ``周辺``
     - ``marginals``
     - 介入の下での出力変数の周辺分布。

因果効果
--------

キーは :doc:`config` にある文字列です（``X → Y=y`` に、``| do(Z)`` / ``| W`` / ``name``）。

.. list-table::
   :header-rows: 1
   :widths: 32 36 32

   * - 日本語
     - 英語
     - 内容
   * - ``名前``
     - ``name``
     - ``name`` を書いたときだけ。キーと同じ文字列。
   * - ``処置``
     - ``treatment``
     - 処置の変数。
   * - ``結果``
     - ``outcome``
     - 結果の変数。
   * - ``結果のラベル``
     - ``outcome_label``
     - ラベル y。
   * - ``処置群``
     - ``treated``
     - 処置群の水準。
   * - ``対照群``
     - ``control``
     - 対照群の水準。
   * - ``P(結果\|do(処置))``
     - ``p_outcome_do``
     - ``{水準: P(Y=y \| do(X=水準))}``。
   * - ``P(結果\|処置)(観測)``
     - ``p_outcome_observed``
     - ``{水準: P(Y=y \| X=水準)}``。P(X=水準) が 0 のところは ``null``。
   * - ``ACE``
     - ``ace``
     - P(y \| do(処置群)) − P(y \| do(対照群))。
   * - ``観測上の差``
     - ``observed_difference``
     - 条件付けでの同じ対比。観測確率が未定義なら ``null``。
   * - ``交絡による差(観測上の差−ACE)``
     - ``confounding_gap``
     - 観測上の差ひく ACE。観測上の差が ``null`` なら ``null``。
   * - ``観測確率0の水準``
     - ``zero_probability_levels``
     - 観測の条件付き確率が未定義な処置の水準。無ければキー自体が無い。
   * - ``層別ACE(given_do)``
     - ``stratified_ace_given_do``
     - ``given_do`` があるとき。
   * - ``条件付きACE(given)``
     - ``conditional_ace_given``
     - ``given`` があるとき。

2 つの層別オブジェクトは同じ形です。

.. list-table::
   :header-rows: 1
   :widths: 32 36 32

   * - 日本語
     - 英語
     - 内容
   * - ``条件``
     - ``given``
     - 層にする変数 id。設定に書いた順。
   * - ``層``
     - ``strata``
     - 層の名前（``Z=z`` または ``Z=z, W=w``）から、その層への写像。
   * - ``層間のACEの差(最後−最初)``
     - ``ace_difference_last_minus_first``
     - 最後の層の ACE ひく最初の層の ACE。その差が未定義なら ``null``。

``given_do`` の層は ``条件の値`` / ``given_values``、
``P(結果|do(処置),do(層))`` / ``p_outcome_do_given_do``、``ACE`` / ``ace`` を持ちます。

``given`` の層は ``条件の値`` / ``given_values``、
``P(層|do(処置))`` / ``p_stratum_do``、
``P(結果|do(処置),層)`` / ``p_outcome_do_given``（do(X) の下でその層の確率が 0 なら ``null``）、
``ACE`` / ``ace``（どちらかの腕が未定義なら ``null``）を持ちます。

感度分析
--------

キーは ``target=label`` です。

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - 日本語
     - 英語
     - 内容
   * - ``目的``
     - ``target``
     - 対象の変数。
   * - ``ラベル``
     - ``label``
     - 対象のラベル。
   * - ``基準のP``
     - ``baseline``
     - 動かす前の P(対象 = ラベル)。
   * - ``変化量delta``
     - ``delta``
     - 使った delta。
   * - ``パラメータ数``
     - ``n_parameters``
     - 上位を残す前に動かしたパラメータの数。
   * - ``上位``
     - ``top``
     - 幅の大きい順の上位の行。

各行のフィールドは、パラメータ名 ``parameter``
（例は ``priors.cloud → Cloudy`` または CPT の行）、
基準値 ``base_value`` 、下げた値 ``low_value`` 、上げた値 ``high_value`` 、
下げたときの確率 ``p_low`` 、上げたときの確率 ``p_high`` 、
幅 ``swing_pt`` （ポイント、小数 2 桁）です。
日本語のキーは ``パラメータ`` 、 ``基準値`` 、 ``下げた値`` 、 ``上げた値`` 、
``P(下げたとき)`` 、 ``P(上げたとき)`` 、 ``幅(pt)`` です。

検算
----

最大差のフィールドは、その比較を実際にしたときだけ数値です。
そうでなければ ``null`` です。Markdown は、最大差 0 とは出さずに、
走らせなかった検算を示します（``未検算`` / ``Not checked``）。
固定の備考は ``備考`` / ``remark`` の
``検算はすべて丸め前の値で比較（出力の確率は小数4桁に丸めている）``
で、キーの言語がどちらでも日本語です。
表示する最大差は有効数字 3 桁です。

.. list-table::
   :header-rows: 1
   :widths: 42 46 12

   * - 日本語
     - 英語
     - ``null`` または無いとき
   * - ``計算モード``
     - ``mode``
     - 無いことはない。``joint`` か ``ancestral``。
   * - ``同時分布の要素数``
     - ``joint_size``
     - 無いことはない。ラベル数の積。
   * - ``einsum(全同時)との最大差``
     - ``max_err_joint_vs_einsum``
     - ``ancestral`` では全同時を作らないので ``null``。
   * - ``einsum(祖先集合・周辺)との最大差``
     - ``max_err_marginal_vs_einsum``
     - 出力の周辺を 1 つも計算しなかったときだけ ``null`` （根以外の変数が無いネットワーク）。
   * - ``einsum(条件付け)との最大差``
     - ``max_err_conditioning_vs_einsum``
     - 観測が無いとき、または同時・ソフトの観測がすべて省略されたとき ``null``（次の行）。単一のハードな観測は ``ancestral`` でも検算する。
   * - ``bayesian_inversionとconditionalizationの最大差``
     - ``max_err_inversion_vs_conditionalization``
     - 1 変数のハードな観測を逆算しなかったとき ``null``。観測が無いときを含む。
   * - ``介入:切断因子分解との最大差``
     - ``max_err_intervention_vs_truncated_factorization``
     - 介入も効果も無ければキー自体が **無い**。do() の周辺を切断因子分解と突き合わせたときは数値。
   * - ``条件付けの検算を省略した観測``
     - ``conditioning_check_skipped``
     - 省略した観測が無ければ **無い**。``ancestral`` では、同時の証拠とソフト証拠をここに挙げ、突き合わせない。
   * - ``bayesian_inversionの検算を省略した観測``
     - ``inversion_check_skipped``
     - 根の全体 × 観測の密テンソルが 100 万要素を超えるとき以外は **無い**。その観測キーを挙げ、逆算の検算は省く。
   * - ``備考``
     - ``remark``
     - ``null`` にはならない。上の日本語の文。

``ancestral`` は、同時分布が 200 万要素を超えると始まります
（テストでは ``run(..., max_joint=...)`` で強制できます）。
条件付けと逆算の検算で使う配線核の上限は 100 万要素です。

.. doctest::

   >>> from fxtensor_bayesnet import run, to_english_keys
   >>> cfg = {
   ...     "as_of": "2026-10-18",
   ...     "variables": {
   ...         "A": {"labels": ["0", "1"]},
   ...         "B": {"labels": ["0", "1"]},
   ...         "C": {"labels": ["0", "1"]},
   ...     },
   ...     "priors": {"A": {"p": [0.6, 0.4]}},
   ...     "kernels": {
   ...         "B": {"parents": ["A"], "cpt": {"0": [0.8, 0.2], "1": [0.3, 0.7]}},
   ...         "C": {"parents": ["B"], "cpt": {"0": [0.9, 0.1], "1": [0.4, 0.6]}},
   ...     },
   ...     "outputs": ["C"],
   ... }
   >>> bare = to_english_keys(run(cfg))["checks"]
   >>> bare["max_err_conditioning_vs_einsum"] is None
   True
   >>> bare["max_err_inversion_vs_conditionalization"] is None
   True
   >>> "conditioning_check_skipped" in bare
   False
   >>> isinstance(bare["max_err_joint_vs_einsum"], float)
   True
   >>> joint = dict(cfg)
   >>> joint["observations"] = [{"name": "both", "evidence": {"B": "1", "C": "1"}, "focus": ["A"]}]
   >>> skipped = run(joint, max_joint=1)["検算"]
   >>> skipped["計算モード"]
   'ancestral'
   >>> skipped["einsum(全同時)との最大差"] is None
   True
   >>> skipped["einsum(条件付け)との最大差"] is None
   True
   >>> skipped["条件付けの検算を省略した観測"]
   ['both']
   >>> en = to_english_keys({"検算": skipped})["checks"]
   >>> en["conditioning_check_skipped"]
   ['both']
   >>> en["max_err_conditioning_vs_einsum"] is None
   True
   >>> one = dict(cfg)
   >>> one["observations"] = [{"var": "C", "label": "1", "focus": ["A"]}]
   >>> still = run(one, max_joint=1)["検算"]
   >>> isinstance(still["einsum(条件付け)との最大差"], float)
   True
   >>> "条件付けの検算を省略した観測" in still
   False

Markdown
--------

``results.md``（と ``--md``）は日本語キーの結果を読み、見出しは ``--lang`` で出します。
シナリオの表は百分率で小数 1 桁です。
基本シナリオの表示は ``基本`` または ``Base`` です。
検算は JSON のあとに、``null`` の最大差と、
``条件付けの検算を省略した観測`` または ``bayesian_inversionの検算を省略した観測`` に入った観測キーを注記します。
英語では、その注記のフィールド名は英語です。

``model.mmd`` は mermaid です。英数字とアンダースコア以外のノード id は短いハッシュに置き、ファイルを妥当に保ちます。
