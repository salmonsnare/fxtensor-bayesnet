fxtensor-bayesnet
==================

This is the English documentation.
The Japanese documentation is the same manual under ``ja/``: `日本語 <ja/index.html>`_.

fxtensor-bayesnet is a config-driven inference tool for discrete Bayesian networks, built on
`fxtensor-salmon <https://github.com/salmonsnare/fxtensor-salmon>`_
(:doc:`documentation <fxtensor-salmon:index>`).
Assumptions and probabilities live in a JSON file, so they can be revised without changing code.
A run compares scenarios, updates beliefs from observations, and can answer causal questions
with the do-operator.

How a model is computed:

- Each variable is discrete and takes one of a finite set of labels, such as ``Clear`` / ``Cloudy``.
- A variable with no parents (a root) has a prior. A child has a conditional probability table:
  each combination of parent labels maps to a distribution over the child (a Markov kernel).
- The joint is built by composing states and kernels with fxtensor-salmon's
  :class:`fxtensor_salmon.FXTensor`.
  From it the tool computes marginals per scenario, conditioning on observations, and
  root-cause posteriors via ``bayesian_inversion``.
- Every run cross-checks library results against a direct NumPy ``einsum``, using unrounded
  values. A mismatch stops the run. See :doc:`output`.
- When the joint would exceed 2,000,000 elements, the tool computes over ancestor sets only.
  The values are the same. Conditioning never builds a dense wiring kernel larger than
  1,000,000 elements; above that it sums the joint or uses variable elimination.
- ``interventions`` apply do() to any variable, root or not, by graph surgery.
  ``effects`` report the average causal effect (ACE) next to the observational difference.
- One observation can combine several pieces of evidence, including soft (likelihood) evidence.
  ``sensitivity`` ranks the prior and CPT entries a probability depends on most.
- ``--plot`` writes the network, scenario comparison, Bayesian updates, and day-to-day changes
  as PNG or SVG. That needs matplotlib.

.. note::

   The probabilities in a config are the user's assumptions.
   The results are consequences of those assumptions.
   The tool does not validate the assumptions themselves.

The package on PyPI is ``fxtensor-bayesnet``. This documentation matches version |release|.
It is licensed under the MIT License.

.. note::

   When an issue is resolved, add a corresponding TikZ figure to these docs.
   Keep the TikZ source and the rendered figure in ``docs/_static/figures/``
   (see ``docs/_static/figures/README.md``) and show the figure in both languages.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   install
   quickstart
   config
   cli
   output
   changelog
   api
