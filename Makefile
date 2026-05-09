.PHONY: test test-fast experiment cv info lint clean help

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  make experiment SPEC=experiments/baseline.yaml  Run one experiment"
	@echo "  make cv         SPEC=experiments/baseline.yaml  Cross-validate"
	@echo "  make info                                       Print package version + surface"
	@echo "  make test                                       Run the full pytest suite"
	@echo "  make test-fast                                  Run contracts only (no integration)"
	@echo "  make lint                                       Ruff + black --check"
	@echo "  make clean                                      Remove caches"
	@echo ""
	@echo "Everything goes through 'wagie experiment …'. No bespoke scripts."

experiment:
	$(PYTHON) -m wagie experiment run $(SPEC)

cv:
	$(PYTHON) -m wagie cv $(SPEC)

info:
	$(PYTHON) -m wagie info

test:
	$(PYTHON) -m pytest -q tests/

test-fast:
	$(PYTHON) -m pytest -q tests/contracts/

cov:
	$(PYTHON) -m pytest -q tests/ --cov=wagie --cov-report=term-missing:skip-covered --cov-branch

cov-html:
	$(PYTHON) -m pytest -q tests/ --cov=wagie --cov-report=html --cov-branch

lint:
	$(PYTHON) -m ruff check src/ tests/ || true
	$(PYTHON) -m black --check src/ tests/ || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .tmp .ruff_cache .mypy_cache
