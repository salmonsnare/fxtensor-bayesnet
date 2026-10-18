Output format
=============

``results.json`` is a JSON object. By default its keys are Japanese, unchanged since 0.9.
``--json-keys en`` and :func:`fxtensor_bayesnet.to_english_keys` write the same tree with
English keys. Names you chose — variable ids, labels, scenario names, intervention names,
effect ``name`` values — are never translated.
An observation key that ends with `` を観測`` loses that suffix; any other observation key
is kept. The remark string inside the checks stays Japanese either way.
Snapshots always store the Japanese keys.

Probabilities in the result are rounded to four decimal places, so a distribution may not
sum to exactly 1. JSON ``null`` is Python ``None``: a comparison that did not run, or a
conditional probability that is undefined. It is not the number 0.

Top level
---------

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Japanese
     - English
     - Contents
   * - ``基準日``
     - ``as_of``
     - The config's ``as_of``.
   * - ``ホライズン``
     - ``horizon``
     - ``horizon``, or ``null``.
   * - ``注意``
     - ``note``
     - ``note``, or ``null``.
   * - ``設定ハッシュ``
     - ``config_hash``
     - First 12 hex characters of SHA-256 over the config JSON with sorted keys.
   * - ``事前分布(基本)``
     - ``priors_base``
     - ``{root: {label: probability}}`` for the base priors.
   * - ``シナリオ``
     - ``scenarios``
     - One entry per scenario, including the base. See below.
   * - ``ベイズ更新(基本)``
     - ``updates_base``
     - One entry per observation, computed under the base priors. Empty object when there are none.
   * - ``介入``
     - ``interventions``
     - Present only when the config has ``interventions``.
   * - ``因果効果``
     - ``effects``
     - Present only when the config has ``effects``.
   * - ``感度分析``
     - ``sensitivity``
     - Present only when the config has ``sensitivity``.
   * - ``検算``
     - ``checks``
     - Cross-check report. Always present.

Scenarios
---------

The base scenario is stored under ``基本`` / ``base``. Other keys are the names from the config.
Each value has:

.. list-table::
   :header-rows: 1
   :widths: 20 22 58

   * - Japanese
     - English
     - Contents
   * - ``上書き``
     - ``overrides``
     - The replacement priors for that scenario (``{root: [probabilities]}``). ``{}`` for the base.
   * - ``周辺``
     - ``marginals``
     - ``{output variable: {label: probability}}``.

Bayesian updates
----------------

The key is ``name`` if the observation has one, otherwise a description plus `` を観測``
(``rain=Some を観測``, or ``rain~L[0.2, 0.9] を観測`` for soft evidence).

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Japanese
     - English
     - Contents
   * - ``証拠``
     - ``evidence``
     - Present when the observation is not a single hard label. Soft evidence is ``{"尤度": [weights]}`` / ``{"likelihood": [weights]}``.
   * - ``観測前の確率``
     - ``evidence_probability``
     - Prior probability of hard evidence, rounded to four decimals. Absent when any evidence is soft.
   * - ``尤度の期待値``
     - ``expected_likelihood``
     - Σ P(x) L(x) when any evidence is soft. Not a probability; may exceed 1. Absent for hard-only evidence.
   * - ``不可能な観測``
     - ``impossible``
     - ``true`` when the observation has probability 0 and the update was skipped. Otherwise the key is absent.
   * - ``事後``
     - ``posterior``
     - ``{variable: {label: probability}}``. Empty when the observation is impossible. Hard-evidence variables are omitted.
   * - ``注目変数の事前``
     - ``focus_prior``
     - Priors of the ``focus`` variables (default: all roots).
   * - ``注目変数の動き``
     - ``focus_movement``
     - Movement for focus variables that are not hard evidence. Empty when the observation is impossible.
   * - ``動きの大きさ順(TV)``
     - ``ranking_by_tv``
     - Those variables, largest total variation first.

Each movement value:

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Japanese
     - English
     - Contents
   * - ``差(pt)``
     - ``delta_pt``
     - Posterior minus prior, in points, one decimal.
   * - ``全変動距離TV``
     - ``tv_distance``
     - Total variation, four decimals.
   * - ``KL(事後\|\|事前)[nat]``
     - ``kl_posterior_prior_nat``
     - KL(posterior \|\| prior) in nats, four decimals.

Interventions
-------------

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Japanese
     - English
     - Contents
   * - ``介入内容``
     - ``do``
     - ``{variable: {label: probability}}`` after expanding a hard label to a one-hot vector.
   * - ``周辺``
     - ``marginals``
     - Marginals of the output variables under the intervention.

Effects
-------

The key is the string documented in :doc:`config` (``X → Y=y``, with ``| do(Z)`` / ``| W`` / ``name``).

.. list-table::
   :header-rows: 1
   :widths: 32 36 32

   * - Japanese
     - English
     - Contents
   * - ``名前``
     - ``name``
     - Present only when ``name`` was set. Same string as the key.
   * - ``処置``
     - ``treatment``
     - Treatment variable.
   * - ``結果``
     - ``outcome``
     - Outcome variable.
   * - ``結果のラベル``
     - ``outcome_label``
     - Label y.
   * - ``処置群``
     - ``treated``
     - Treated level.
   * - ``対照群``
     - ``control``
     - Control level.
   * - ``P(結果\|do(処置))``
     - ``p_outcome_do``
     - ``{level: P(Y=y \| do(X=level))}``.
   * - ``P(結果\|処置)(観測)``
     - ``p_outcome_observed``
     - ``{level: P(Y=y \| X=level)}``. ``null`` where P(X=level) is 0.
   * - ``ACE``
     - ``ace``
     - P(y \| do(treated)) − P(y \| do(control)).
   * - ``観測上の差``
     - ``observed_difference``
     - The same contrast under conditioning. ``null`` if either observed probability is undefined.
   * - ``交絡による差(観測上の差−ACE)``
     - ``confounding_gap``
     - Observed difference minus ACE. ``null`` when the observed difference is ``null``.
   * - ``観測確率0の水準``
     - ``zero_probability_levels``
     - Treatment levels where the observed conditional is undefined. Absent when there are none.
   * - ``層別ACE(given_do)``
     - ``stratified_ace_given_do``
     - Present when ``given_do`` was set.
   * - ``条件付きACE(given)``
     - ``conditional_ace_given``
     - Present when ``given`` was set.

Both stratification objects have the same shape:

.. list-table::
   :header-rows: 1
   :widths: 32 36 32

   * - Japanese
     - English
     - Contents
   * - ``条件``
     - ``given``
     - The stratification variable ids, in config order.
   * - ``層``
     - ``strata``
     - Map of stratum name (``Z=z`` or ``Z=z, W=w``) to one stratum.
   * - ``層間のACEの差(最後−最初)``
     - ``ace_difference_last_minus_first``
     - ACE of the last stratum minus ACE of the first. ``null`` if that difference is undefined.

A ``given_do`` stratum contains ``条件の値`` / ``given_values``,
``P(結果|do(処置),do(層))`` / ``p_outcome_do_given_do``, and ``ACE`` / ``ace``.

A ``given`` stratum contains ``条件の値`` / ``given_values``,
``P(層|do(処置))`` / ``p_stratum_do``,
``P(結果|do(処置),層)`` / ``p_outcome_do_given`` (``null`` where the stratum has probability 0 under do(X)),
and ``ACE`` / ``ace`` (``null`` when either arm is undefined).

Sensitivity
-----------

The key is ``target=label``.

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Japanese
     - English
     - Contents
   * - ``目的``
     - ``target``
     - Target variable.
   * - ``ラベル``
     - ``label``
     - Target label.
   * - ``基準のP``
     - ``baseline``
     - P(target = label) before perturbation.
   * - ``変化量delta``
     - ``delta``
     - The delta that was used.
   * - ``パラメータ数``
     - ``n_parameters``
     - How many parameters were perturbed, before keeping the top rows.
   * - ``上位``
     - ``top``
     - The top rows, largest swing first.

Each row: ``パラメータ`` / ``parameter`` (for example ``priors.cloud → Cloudy`` or a CPT row),
``基準値`` / ``base_value``, ``下げた値`` / ``low_value``, ``上げた値`` / ``high_value``,
``P(下げたとき)`` / ``p_low``, ``P(上げたとき)`` / ``p_high``, and
``幅(pt)`` / ``swing_pt`` (two decimal places).

Checks
------

A max-error field is a number only when that comparison ran.
Otherwise it is ``null``. Markdown then says which checks were not run
(``未検算`` / ``Not checked``), instead of showing a max error of 0.
The fixed remark is
``検算はすべて丸め前の値で比較（出力の確率は小数4桁に丸めている）``
under ``備考`` / ``remark``, in Japanese in both key languages.
Displayed max errors use three significant digits.

.. list-table::
   :header-rows: 1
   :widths: 42 46 12

   * - Japanese
     - English
     - When it is ``null`` or absent
   * - ``計算モード``
     - ``mode``
     - Never. ``joint`` or ``ancestral``.
   * - ``同時分布の要素数``
     - ``joint_size``
     - Never. Product of the label counts.
   * - ``einsum(全同時)との最大差``
     - ``max_err_joint_vs_einsum``
     - ``null`` in ``ancestral`` mode, because no full joint is built.
   * - ``einsum(祖先集合・周辺)との最大差``
     - ``max_err_marginal_vs_einsum``
     - ``null`` only when no output marginal was computed (a network with no non-root variables).
   * - ``einsum(条件付け)との最大差``
     - ``max_err_conditioning_vs_einsum``
     - ``null`` when there are no observations, or when every joint/soft observation was skipped (see the next row). A single hard observation is still checked in ``ancestral`` mode.
   * - ``bayesian_inversionとconditionalizationの最大差``
     - ``max_err_inversion_vs_conditionalization``
     - ``null`` when no single-variable hard observation was inverted, including when there are no observations.
   * - ``介入:切断因子分解との最大差``
     - ``max_err_intervention_vs_truncated_factorization``
     - The key is **absent** when the run had no interventions and no effects. A number when those do() marginals were cross-checked against truncated factorization.
   * - ``条件付けの検算を省略した観測``
     - ``conditioning_check_skipped``
     - **Absent** unless some observation was skipped. In ``ancestral`` mode, joint evidence and soft evidence are listed here and are not cross-checked.
   * - ``bayesian_inversionの検算を省略した観測``
     - ``inversion_check_skipped``
     - **Absent** unless the dense roots × observation tensor would exceed 1,000,000 elements. Those observation keys are listed and the inversion check is skipped.
   * - ``備考``
     - ``remark``
     - Never null. The Japanese sentence above.

``ancestral`` mode starts when the joint has more than 2,000,000 elements
(``run(..., max_joint=...)`` can force it in tests).
The wiring-kernel limit for conditioning and for the inversion cross-check is 1,000,000 elements.

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

``results.md`` (and ``--md``) reads the Japanese-key result and prints headings in ``--lang``.
Scenario tables use percentages to one decimal.
The base scenario is labelled ``基本`` or ``Base``.
Checks are printed as JSON, then a note listing max-error fields that are ``null``
and any observation keys in ``条件付けの検算を省略した観測`` or
``bayesian_inversionの検算を省略した観測``.
In English that note uses the English field names.

``model.mmd`` is mermaid. A node id that is not letters, digits, and underscores is replaced
with a short hash so the file stays valid.
