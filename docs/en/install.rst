Installation
============

Python 3.8 or later. The runtime dependencies are NumPy and
`fxtensor-salmon <https://github.com/salmonsnare/fxtensor-salmon>`_ (``fxtensor-salmon>=1.2.2,<2``).
matplotlib is not required unless you draw figures.

.. code-block:: bash

   pip install fxtensor-bayesnet
   pip install "fxtensor-bayesnet[plot]"

The console script is ``fxtensor-bayesnet``. ``python -m fxtensor_bayesnet`` is the same program.

Development
-----------

.. code-block:: bash

   git clone https://github.com/salmonsnare/fxtensor-bayesnet.git
   cd fxtensor-bayesnet
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -e ".[dev]"
   pytest

``[dev]`` installs pytest and matplotlib. It does not add any dependency to a normal install.

Building these docs
-------------------

The documentation extra needs Python 3.10 or later (Sphinx 8).
The library itself still runs on Python 3.8.

.. code-block:: bash

   pip install -e ".[docs]"
   # or: pip install -r docs/requirements.txt && pip install -e .
   make -C docs html doctest

``make html`` writes English pages to ``docs/_build/html/`` and Japanese pages to
``docs/_build/html/ja/``. Warnings are treated as errors.
``make doctest`` runs the examples embedded in both languages.

Open ``docs/_build/html/index.html``. The same site is published from the ``main`` branch to
https://salmonsnare.github.io/fxtensor-bayesnet/ when GitHub Pages is set to **GitHub Actions**
as its source. See the workflow ``.github/workflows/docs.yml``.
