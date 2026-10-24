# Everything here runs without a GPU. The GPU work lives in the notebooks.

PYTHON ?= python3

.PHONY: help check test notebooks site html responsive parity serve aggregate clean

help:
	@echo "make check      run every local check"
	@echo "make test       harness + kernel-math tests"
	@echo "make notebooks  regenerate notebooks/*.ipynb from tools/nbsrc/"
	@echo "make html       site structure, anchors, assets, a11y (no browser needed)"
	@echo "make site       the website's roofline / KV maths (needs node)"
	@echo "make responsive no horizontal scroll at phone width (needs Chrome)"
	@echo "make parity     check Python and JavaScript agree on every formula"
	@echo "make aggregate  results/ -> docs/data/results.json"
	@echo "make serve      serve docs/ at http://localhost:8080"
	@echo ""
	@echo "Set REPO_SLUG=owner/name so Colab badges and site links point at your fork."

check: test notebooks html site parity responsive
	@echo ""
	@echo "All local checks passed."

html:
	$(PYTHON) tools/check_html.py

responsive:
	@command -v node >/dev/null 2>&1 \
		&& node tools/check_responsive.mjs \
		|| echo "node not found -- skipping the responsive check"

test:
	$(PYTHON) tests/test_bench.py
	@echo ""
	$(PYTHON) tests/test_kernel_math.py

notebooks:
	$(PYTHON) tools/build_notebooks.py

site:
	@command -v node >/dev/null 2>&1 \
		&& node tools/check_site.mjs \
		|| echo "node not found -- skipping site math check"

parity:
	$(PYTHON) tools/check_parity.py

aggregate:
	$(PYTHON) -m bench.aggregate

serve: aggregate
	@echo "http://localhost:8080"
	@cd docs && $(PYTHON) -m http.server 8080

clean:
	rm -rf logs/ .parity_dump.mjs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
