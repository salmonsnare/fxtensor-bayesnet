Quickstart
==========

The model below is a two-variable weather sketch. Labels and probabilities are illustrative.
``cloud`` is a root. ``rain`` depends only on ``cloud``:

- P(rain = Some | Clear) = 0.1
- P(rain = Some | Cloudy) = 0.6
- P(cloud = Clear) = 0.6, so P(rain = Some) = 0.6 × 0.1 + 0.4 × 0.6 = 0.3
- Scenario ``A: Mostly cloudy`` replaces the cloud prior with ``[0.2, 0.8]``,
  so P(rain = Some) = 0.2 × 0.1 + 0.8 × 0.6 = 0.5
- After observing rain = Some, P(cloud = Cloudy | rain = Some) = (0.6 × 0.4) / 0.3 = 0.8

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

``run`` returns Japanese keys. :func:`fxtensor_bayesnet.to_english_keys` translates the fixed
keys and leaves the names you chose (``cloud``, ``Clear``, ``A: Mostly cloudy``) as they are.
The base scenario's Japanese key ``基本`` becomes ``base``.

Command line
------------

Save the same JSON as ``config.json`` and validate it. ``--check`` does not write results.

.. code-block:: bash

   fxtensor-bayesnet --check
   fxtensor-bayesnet --no-write --md --lang en
   fxtensor-bayesnet --no-write --json-keys en

``--check`` on a valid file exits 0. A missing file, invalid JSON, or a config error exits 2.

Larger examples in the repository
---------------------------------

``examples/`` is not part of the PyPI distribution. Fetch it from the
`repository <https://github.com/salmonsnare/fxtensor-bayesnet>`_.

- ``examples/weather/`` — pressure, humidity, cloud, rain, an outdoor event, and traffic.
  Japanese ``config.json`` and English ``config.en.json`` use the same numbers.
- ``examples/maintenance/`` — load, age, maintenance, wear, vibration, bearing temperature,
  and failure. Again in both languages.
- ``examples/ecology_host_parasitoid/`` — English only. Interventions, effects, joint and
  soft evidence, and sensitivity. Every number there is an illustrative placeholder.

Figures for those examples are regenerated with ``python docs/make_images.py``
(needs the ``plot`` extra) into ``docs/images/`` and ``docs/images/en/``.
