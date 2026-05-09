.PHONY: test test-fast round preflight fast-train clean lint help loop-status loop-daemon critic compact compact-apply merge-round kill-round

PYTHON ?= python

help:
	@echo "Targets:"
	@echo "  make test          Run the full pytest suite"
	@echo "  make test-fast     Run only the causality + weights tests (fast subset)"
	@echo "  make preflight     Run scripts/run_round.py --preflight"
	@echo "  make round         Open a fresh round branch (delegates to the agent)"
	@echo "  make fast-train    Train a CatBoost run with FAST_MODE=1 (year 2024 only)"
	@echo "  make lint          Ruff + black --check (no autoformat)"
	@echo "  make clean         Remove __pycache__, .pytest_cache, mlruns/.trash, .tmp"
	@echo "  make loop-status   Print loop health digest (heartbeat + LEDGER tail + branches)"
	@echo "  make loop-daemon   Run scripts/loop_daemon.ps1 (PowerShell, foreground)"
	@echo "  make critic        Run CRITIC checklist on current branch"
	@echo "  make compact       Dry-run compaction (LEDGER, plots, branches, MLflow)"
	@echo "  make compact-apply Apply compaction"
	@echo "  make merge-round H=H-005    FF current agent/round-* branch onto master + tag + delete"
	@echo "  make kill-round H=H-005 R='reason'  Delete current branch + append KILL_LIST"

loop-status:
	$(PYTHON) scripts/loop_status.py

loop-tail:
	$(PYTHON) scripts/tail_round.py

loop-tail-tools:
	$(PYTHON) scripts/tail_round.py --tools-only --tail 60

loop-daemon:
	powershell -ExecutionPolicy Bypass -File scripts/loop_daemon.ps1

loop-daemon-dry:
	powershell -ExecutionPolicy Bypass -File scripts/loop_daemon.ps1 -DryRun -MaxRounds 2

loop-once:
	powershell -ExecutionPolicy Bypass -File scripts/loop_daemon.ps1 -MaxRounds 1

critic:
	$(PYTHON) scripts/critic_check.py

compact:
	$(PYTHON) scripts/compact_loop_state.py

compact-apply:
	$(PYTHON) scripts/compact_loop_state.py --apply

merge-round:
	$(PYTHON) scripts/merge_round.py merge --hypothesis $(H)

kill-round:
	$(PYTHON) scripts/merge_round.py kill --hypothesis $(H) --reason "$(R)"

test:
	$(PYTHON) -m pytest -q tests/

test-fast:
	$(PYTHON) -m pytest -q tests/test_causality.py tests/test_weights.py tests/test_splits.py

preflight:
	$(PYTHON) scripts/run_round.py --preflight

round: preflight
	@echo "Round preflight passed. The autonomous agent now picks up the next BACKLOG item."
	@echo "To trigger manually, run the scheduled task once via the agent harness."

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
