"""Tests for src/training_helpers.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import training_helpers


def test_param_provenance_pinned_takes_priority():
    prov = training_helpers.param_provenance(
        use_pinned=True,
        run_hpo=True,
        pinned_params={"a": 1},
        hpo_best_params={"a": 2},
        file_loaded_params={"a": 3},
    )
    assert prov["source"] == "pinned"
    assert prov["params"] == {"a": 1}


def test_param_provenance_hpo_when_no_pin():
    prov = training_helpers.param_provenance(
        use_pinned=False,
        run_hpo=True,
        pinned_params={"a": 1},
        hpo_best_params={"a": 2},
        file_loaded_params={"a": 3},
    )
    assert prov["source"] == "hpo"
    assert prov["params"] == {"a": 2}


def test_param_provenance_file_when_no_hpo():
    prov = training_helpers.param_provenance(
        use_pinned=False,
        run_hpo=False,
        pinned_params=None,
        hpo_best_params=None,
        file_loaded_params={"a": 3},
    )
    assert prov["source"] == "file"
    assert prov["params"] == {"a": 3}


def test_param_provenance_fallback():
    prov = training_helpers.param_provenance(
        use_pinned=False, run_hpo=False, file_loaded_params=None
    )
    assert prov["source"] == "fallback"
    assert prov["params"] == {}


def test_scalarize_handles_common_types():
    assert training_helpers._scalarize(42) == 42
    assert training_helpers._scalarize(0.5) == 0.5
    assert training_helpers._scalarize(True) is True
    assert training_helpers._scalarize("hi") == "hi"
    assert training_helpers._scalarize(None) == "None"
    assert training_helpers._scalarize([1, 2]) == "[1, 2]"
    # NaN/Inf via repr (MLflow accepts NaN as float but the repr fallback is safer)
    assert training_helpers._scalarize(float("nan")) == "nan"
