Configuration reference
=======================

A config is one JSON object. ``load_config`` reads it and :func:`fxtensor_bayesnet.config.validate`
checks it. ``--check`` prints every problem it finds and exits with code 2.
Inference does not start.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Key
     - Required
     - Meaning
   * - ``as_of``
     - yes
     - Reference date, conventionally ``YYYY-MM-DD``. The snapshot folder is named with this
       string. ``--check`` does not parse it as a date.
       Automatic ``--diff`` only considers folders whose names match ``YYYY-MM-DD`` and are
       strictly less than ``as_of`` as strings.
   * - ``horizon``
     - 
     - Shown in the results (``ホライズン`` / ``horizon``). Omitted means JSON ``null``.
   * - ``note``
     - 
     - Shown as a note in the results (``注意`` / ``note``). Omitted means JSON ``null``.
   * - ``variables``
     - yes
     - Map of variable id to ``{"name": display name, "labels": [labels, ...]}``.
       ``name`` is optional; Markdown then uses the id.
       At least two labels, no duplicates. Order in this object is free: roots need not come first.
   * - ``priors``
     - yes
     - Root id to ``{"p": [probability per label], "reason": optional text}``.
       ``p`` has one entry per label, no negatives, and sums to 1 within ``1e-6``.
   * - ``kernels``
     - yes
     - Child id to ``{"parents": [parent ids], "cpt": {...}, "reason": optional text}``.
   * - ``scenarios``
     - 
     - Scenario name to ``{root id: replacement prior}``.
       The base scenario is always added and is not written here.
       A non-root is an error; the message points at ``interventions``.
   * - ``observations``
     - 
     - List of observations. See below.
   * - ``interventions``
     - 
     - Intervention name to ``{variable: label or distribution}``. See below.
   * - ``effects``
     - 
     - List of causal-effect questions. See below.
   * - ``sensitivity``
     - 
     - One object, or a list of them. See below.
   * - ``outputs``
     - 
     - Variable ids shown in result tables.
       Omitted, or an empty list, means every non-root.
       A non-root left out of ``outputs`` is an intermediate variable (grey on the diagram).

Any other top-level key is not validated. It stays in the file, so it changes
:func:`fxtensor_bayesnet.config.config_hash`, and inference does not read it.
``reason`` is stored for priors and kernels and is reported by ``--diff`` when it changes.
It is not copied into ``results.json``.

Each variable appears in exactly one of ``priors`` or ``kernels``.
The parent relation must be acyclic.
At most 52 variables: :class:`fxtensor_bayesnet.network.Net` raises ``ValueError`` above that.
``--check`` does not apply this limit, because it never builds the network.

Conditional probability tables
------------------------------

Each CPT key joins the parents' labels with ``|``, in ``parents`` order.
With ``"parents": ["load", "age"]`` a row key looks like ``"High|Old"``.
Each value is a list of probabilities in the child's label order. Every row sums to 1
within ``1e-6``. Every combination of parent labels needs a row.
``--check`` reports a missing row and an extra row separately.

Observations
------------

Two forms. They cannot be mixed in one item.

Hard evidence on one variable:

.. code-block:: json

   {"var": "rain", "label": "Some", "focus": ["cloud"]}

``focus`` lists variables whose prior-to-posterior movement is reported
(difference in points, total variation, KL in nats) and ranked by total variation.
Omitted ``focus`` means every root. Unknown ids are errors.

Joint evidence, and soft evidence:

.. code-block:: json

   {"name": "Sensor suggests Some",
    "evidence": {"rain": {"likelihood": [0.2, 0.9]}},
    "focus": ["cloud"]}

- A string value is hard evidence (that label).
- ``{"likelihood": [...]}`` is soft evidence: a non-negative weight per label,
  ``P(report | true label)``. The weights need not sum to 1, and at least one must be positive.
  The length must match the labels.
- Every entry in ``evidence`` is applied together.
- ``name``, when set, is the result key. Otherwise the key is built from the evidence,
  for example ``rain=Some を観測`` or ``rain~L[0.2, 0.9] を観測``.
- With ``--json-keys en``, a key that ends in `` を観測`` drops that suffix.
  A ``name`` you wrote is never translated.

An observation whose probability is 0 under the model does not stop the run.
``run`` warns, skips that update, and records it with ``不可能な観測`` / ``impossible`` true,
probability 0, and an empty posterior. Other observations are computed as usual.
``updates.png`` leaves the impossible one out.
Calling :meth:`Engine.condition_raw <fxtensor_bayesnet.inference.Engine.condition_raw>` or
:meth:`Engine.posterior_raw <fxtensor_bayesnet.inference.Engine.posterior_raw>` directly still
raises ``ValueError`` in that case.

When every piece of evidence is hard, the update stores ``観測前の確率`` / ``evidence_probability``,
the prior probability of the evidence.
As soon as one piece is soft, it stores ``尤度の期待値`` / ``expected_likelihood`` instead:
the expectation of the likelihood weights, Σ P(x) L(x). That is not a probability.
It can exceed 1, and Markdown shows it as a number rather than a percentage.

Interventions
-------------

.. code-block:: json

   "interventions": {
     "Force cloudy": {"cloud": "Cloudy"},
     "Half cloudy": {"cloud": [0.5, 0.5]}
   }

A label is a hard do(). A list is a stochastic intervention and must be a distribution
(length, non-negative, sums to 1). Several variables in one object are intervened on together.
Incoming edges of each intervened variable are cut and everything downstream is recomputed.
Upstream variables keep their priors. That is the difference from conditioning.

``scenarios`` remain the way to replace a root prior.
do() on a root matches a scenario that sets the same distribution.
The result reports marginals of ``outputs`` under the intervention.

Effects
-------

Each item asks for P(outcome | do(treatment)) and P(outcome | treatment) at every treatment
level, plus the ACE, the observational difference, and their gap.

.. code-block:: json

   {"treatment": "cloud", "outcome": "rain", "outcome_label": "Some",
    "treated": "Cloudy", "control": "Clear"}

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``treatment``
     - Variable X. Required. Must differ from ``outcome``.
   * - ``outcome``
     - Variable Y. Required.
   * - ``outcome_label``
     - Label y of Y. Default: the last label of Y.
   * - ``treated``
     - Level of X used as the treated arm. Default: the last label of X.
   * - ``control``
     - Level of X used as the control arm. Default: the first label of X.
       After defaults, ``treated`` and ``control`` must differ.
   * - ``given_do``
     - Optional ``{Z: labels}``. Z is also fixed by intervention. Stratified ACE.
   * - ``given``
     - Optional ``{W: labels}``. W is observed in the intervened model. Conditional ACE.
   * - ``name``
     - Optional non-empty string. Used as the result key, and stored as ``名前`` / ``name``.

A value of ``given_do`` or ``given`` may be a list of labels, one label, or ``null`` (every label).
Several variables produce the Cartesian product of their labels. A stratum is named
``Z=z`` or ``Z=z, W=w``. Z must not be the treatment or the outcome, and a variable cannot
appear in both keys. Unknown variables, unknown labels, and duplicate labels are config errors.

For each stratum z of ``given_do``, the result has P(Y=y | do(X=x), do(Z=z)) at every level x
and ACE(z) = P(y | do(treated), do(z)) − P(y | do(control), do(z)).
For ``given``, X is fixed by intervention and Z is conditioned on in that model:
P(Y=y | do(X=x), Z=z). A stratum with probability 0 under do(X=x) is undefined
(JSON ``null``, Markdown ``—``) and ``run`` warns.
The difference between the last stratum's ACE and the first's is stored as
``層間のACEの差(最後−最初)`` / ``ace_difference_last_minus_first``
(``null`` when there is only one stratum, or when either ACE is undefined).

Result key
~~~~~~~~~~

:func:`fxtensor_bayesnet.causal.effect_key` builds the key:

- ``name`` if it is set.
- Otherwise ``X → Y=y``.
- With ``given_do`` only: ``X → Y=y | do(Z)``. Several variables: ``X → Y=y | do(Z, W)``,
  in the order of the object.
- With ``given`` only: ``X → Y=y | W``.
- With both: ``X → Y=y | do(Z), W``.

The labels chosen inside ``given_do`` / ``given`` are not part of the key, only the variable ids.
Effects with no stratification keep the plain ``X → Y=y`` key, including effects written before
these fields existed.

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

Duplicate keys
~~~~~~~~~~~~~~

Two items that would share a key are a config error. A typical case is the same
``given_do`` variable with different label lists: both keys are ``X → Y=y | do(Z)``.
``--check`` reports it and exits with code 2.
:func:`fxtensor_bayesnet.config.validate` and :func:`fxtensor_bayesnet.run` raise
:class:`~fxtensor_bayesnet.ConfigError` instead of keeping only the last item.
Give the items different ``name`` values, or change the stratification variables.
The exception text is Japanese and contains ``重なる``.

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

If a treatment level has probability 0, P(outcome | treatment = that level) is undefined.
The JSON value is ``null``, the level is listed in ``観測確率0の水準`` / ``zero_probability_levels``,
and ``run`` warns. The do() probabilities and the ACE stay defined.
Markdown shows ``—``. ``effects.png`` draws no observed bar for that level and marks it
"undefined (P=0)".

Sensitivity
-----------

``sensitivity`` is one object or a list:

.. code-block:: json

   {"target": "rain", "label": "Some", "delta": 0.05, "top": 10}

``label`` defaults to the last label of ``target``.
``delta`` defaults to 0.05 and must be strictly between 0 and 1.
``top`` defaults to 10 and must be an integer of at least 1.

Each parameter in the ancestor set of the target — one prior entry, or one entry of one CPT
row — is moved by ±delta, clipped to [0, 1]. The other entries of that row are rescaled
proportionally so the row still sums to 1. A row with two labels moves only the last label.
Results are sorted by swing (the absolute change in P(target = label), in points).
The stored list is the top ``top`` rows. The parameter count is the count before that cut.
``--plot`` draws a tornado chart.

What ``--check`` rejects
------------------------

``validate`` collects every problem and raises :class:`~fxtensor_bayesnet.ConfigError`.
The message starts with ``設定エラー:`` and is Japanese.
``--check`` prints it and exits with code 2. A valid file prints ``設定OK`` and exits 0,
and does not write results.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Situation
     - Result
   * - ``as_of`` missing
     - ConfigError
   * - ``variables`` empty
     - ConfigError
   * - Fewer than two labels, or a duplicate label
     - ConfigError
   * - A variable is in both ``priors`` and ``kernels``, or in neither
     - ConfigError
   * - A prior or kernel names a variable that is not in ``variables``
     - ConfigError
   * - A distribution has the wrong length, a negative entry, or does not sum to 1 (tolerance ``1e-6``)
     - ConfigError
   * - A parent is undefined
     - ConfigError
   * - A CPT row is missing or extra
     - ConfigError
   * - A scenario replaces a non-root
     - ConfigError (the message mentions ``interventions``)
   * - An observation is not an object, the label is unknown, or ``evidence`` is combined with ``var`` / ``label``
     - ConfigError
   * - A likelihood has the wrong length, a negative weight, or no positive weight
     - ConfigError
   * - ``focus`` names an unknown variable
     - ConfigError
   * - An intervention is empty, names an unknown variable, or is not a label or a distribution
     - ConfigError
   * - An effect is missing ``treatment`` / ``outcome``, uses one variable for both, cites an unknown label, or has ``treated`` equal to ``control``
     - ConfigError
   * - ``given_do`` / ``given`` is malformed, names the treatment or outcome, repeats a label, or lists one variable in both
     - ConfigError
   * - ``name`` is present but is not a non-empty string
     - ConfigError
   * - Two effects would share a result key
     - ConfigError from ``validate`` and from ``run``; ``--check`` exits 2
   * - ``sensitivity`` has an unknown target or label, a ``delta`` outside (0, 1), or a ``top`` that is not an integer ≥ 1
     - ConfigError
   * - ``outputs`` names an unknown variable
     - ConfigError
   * - The graph has a cycle
     - ConfigError
   * - More than 52 variables
     - ``ValueError`` when the network is built, not from ``--check``
   * - The file is missing or is not JSON
     - CLI exit code 2 (not ``ConfigError``)

A performed cross-check that disagrees with the direct computation raises ``AssertionError``
from ``run`` and stops the process. Tolerances are ``1e-12`` for the full joint, the
marginals, and interventions, and ``1e-9`` for conditioning and for ``bayesian_inversion``.
