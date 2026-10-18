# Build the bilingual docs into docs/_build/html (English at the root, Japanese under ja/).
PYTHON ?= python3
SPHINX ?= $(PYTHON) -m sphinx
OPTS ?= -W --keep-going

.PHONY: html doctest clean

html:
	$(SPHINX) $(OPTS) -b html en _build/html
	$(SPHINX) $(OPTS) -b html ja _build/html/ja

doctest:
	$(SPHINX) $(OPTS) -b doctest en _build/doctest/en
	$(SPHINX) $(OPTS) -b doctest ja _build/doctest/ja

clean:
	rm -rf _build
