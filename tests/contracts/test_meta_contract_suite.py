"""Meta-test: assert that the contract test suite itself is intact.

Without this, a refactor that quietly deletes
`test_predict_then_learn_ordering_when_label_matures` (or any other named
invariant) would land with green CI. This test fails when:

    1. The contract test suite shrinks below `MIN_CONTRACT_TESTS`.
    2. A required test FILE is missing from `tests/contracts/`.
    3. A required test FUNCTION (the PINNED_TESTS list) has been deleted /
       renamed.

The test discovers tests via pytest's collection API directly on the
contracts directory, so it sees what CI would actually run.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest


CONTRACTS_DIR = Path(__file__).parent

# Floor: the contract suite has 530+ tests at the time this lands. Setting
# the floor to 30 catches anyone wholesale-deleting the suite without making
# this test brittle to legitimate single-test removals during refactors.
MIN_CONTRACT_TESTS = 30

# These FILES are load-bearing; deleting any of them removes core invariants.
REQUIRED_FILES = (
    "test_pipeline_construction.py",
    "test_engine_invariants.py",
    "test_label_buffer.py",
    "test_no_leakage_canary.py",        # T7 + T7-meta
    "test_meta_contract_suite.py",      # this file (self-reference)
    "test_pipeline_state.py",
    "test_features_streaming.py",
    "test_online_arf.py",
    "test_metrics_battery.py",
    "test_metrics_calibration.py",
)

# These TEST FUNCTIONS are pinned. If any is deleted or renamed, this test
# fires with the missing name. To intentionally remove one, edit the list AND
# document the removal in the same commit.
PINNED_TESTS = (
    # T7 canary
    ("test_no_leakage_canary.py",
     "test_T7_canary_no_leakage_when_cheat_column_present_but_not_in_features"),
    ("test_no_leakage_canary.py",
     "test_T7_meta_canary_fires_when_cheat_is_injected_into_features"),
    ("test_no_leakage_canary.py",
     "test_T7_audit_captures_p_online_at_every_decision"),
    # Pipeline construction guards
    ("test_pipeline_construction.py", "test_pipeline_requires_strategy_at_end"),
    ("test_pipeline_construction.py", "test_pipeline_rejects_out_of_order_kinds"),
    ("test_pipeline_construction.py", "test_pipeline_accepts_valid_order_no_calibrator"),
    ("test_pipeline_construction.py", "test_pipeline_rejects_duplicate_names"),
    # Predict-then-learn ordering (engine invariant I.7)
    ("test_engine_invariants.py",
     "test_predict_then_learn_ordering_when_label_matures"),
    ("test_engine_invariants.py", "test_warmup_emits_zero_decisions"),
    # Label buffer maturation
    ("test_label_buffer.py",
     "test_maybe_emit_label_one_when_excursion_meets_alpha"),
    ("test_label_buffer.py",
     "test_maybe_emit_label_zero_when_alpha_too_large"),
    ("test_label_buffer.py", "test_maybe_emit_segment_boundary_suppression"),
)


def _list_test_function_names(file_path: Path) -> set[str]:
    """Parse a python test file and return all `def test_*` names without
    importing it (avoid module-level fixture collisions)."""
    names: set[str] = set()
    src = file_path.read_text(encoding="utf-8")
    for m in re.finditer(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(", src, re.MULTILINE):
        names.add(m.group(1))
    return names


def test_contract_suite_has_minimum_tests():
    """The total number of `def test_*` functions in tests/contracts/ must
    not drop below MIN_CONTRACT_TESTS. This is the floor against silent
    suite-deletion."""
    total = 0
    for f in CONTRACTS_DIR.glob("test_*.py"):
        total += len(_list_test_function_names(f))
    assert total >= MIN_CONTRACT_TESTS, (
        f"contract suite has {total} tests; minimum is {MIN_CONTRACT_TESTS}. "
        "If a removal is intentional, lower MIN_CONTRACT_TESTS in this file."
    )


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_required_file_exists(filename: str):
    """Each REQUIRED_FILES entry must exist in tests/contracts/."""
    p = CONTRACTS_DIR / filename
    assert p.is_file(), (
        f"required contract file missing: {filename}. If this removal is "
        f"intentional, edit REQUIRED_FILES in {Path(__file__).name} and "
        "document why in the commit message."
    )


@pytest.mark.parametrize("filename, test_name", PINNED_TESTS)
def test_pinned_test_function_exists(filename: str, test_name: str):
    """Each (file, test_name) in PINNED_TESTS must still be present."""
    p = CONTRACTS_DIR / filename
    if not p.is_file():
        pytest.fail(
            f"pinned test {test_name} cannot exist — its file {filename} is missing"
        )
    names = _list_test_function_names(p)
    assert test_name in names, (
        f"pinned test {test_name!r} missing from {filename}. "
        "If renaming, update PINNED_TESTS in test_meta_contract_suite.py "
        "in the SAME commit."
    )
