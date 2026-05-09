.PHONY: test test-fast preflight fast-train clean lint help critic compact compact-apply

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  make test          Run the full pytest suite"
	@echo "  make test-fast     Run only the causality + weights tests (fast subset)"
	@echo "  make preflight     Run scripts/run_round.py --preflight"
	@echo "  make critic        Run CRITIC checklist on staged + working changes"
	@echo "  make compact       Dry-run compaction (LEDGER, plots, MLflow)"
	@echo "  make compact-apply Apply compaction"
	@echo "  make fast-train    Train a CatBoost run with FAST_MODE=1 (year 2024 only)"
	@echo "  make lint          Ruff + black --check (no autoformat)"
	@echo "  make clean         Remove __pycache__, .pytest_cache, mlruns/.trash, .tmp"
	@echo ""
	@echo "Loop runs in this Claude Code session (see RESEARCH/LOOP_DISCIPLINE.md)."

critic:
	$(PYTHON) scripts/critic_check.py

compact:
	$(PYTHON) scripts/compact_loop_state.py

compact-apply:
	$(PYTHON) scripts/compact_loop_state.py --apply

test:
	$(PYTHON) -m pytest -q tests/

test-fast:
	$(PYTHON) -m pytest -q tests/test_causality.py tests/test_weights.py tests/test_splits.py

preflight:
	$(PYTHON) scripts/run_round.py --preflight

fast-train:
	BARRIER_FAST_MODE=1 $(PYTHON) -c "import nbformat; from nbconvert.preprocessors import ExecutePreprocessor; \
		nb = nbformat.read('notebooks/03_model_training.ipynb', as_version=4); \
		ExecutePreprocessor(timeout=1800).preprocess(nb, {'metadata': {'path': 'notebooks/'}})"

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/ || true
	$(PYTHON) -m black --check src/ tests/ scripts/ || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .tmp .ruff_cache .mypy_cache
	rm -rf experiments/mlruns/.trash 2>/dev/null || true
