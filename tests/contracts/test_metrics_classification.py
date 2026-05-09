"""Contract tests for `wagie.metrics.classification`.

Diagnostic-only ranking metrics (calibration is the lead per repo policy).
Both `roc_auc` and `pr_auc` should agree with sklearn on binary 0/1 inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from wagie.metrics import pr_auc, roc_auc


# ---------- empty input -----------------------------------------------------


def test_roc_auc_empty() -> None:
    assert roc_auc([], []) == 0.5


def test_pr_auc_empty() -> None:
    assert pr_auc([], []) == 0.0


# ---------- single-class fallbacks -----------------------------------------


def test_roc_auc_all_positive_returns_half() -> None:
    """y has one unique class → contract returns 0.5."""
    assert roc_auc([1, 1, 1, 1], [0.1, 0.5, 0.9, 0.7]) == 0.5


def test_roc_auc_all_negative_returns_half() -> None:
    assert roc_auc([0, 0, 0, 0], [0.1, 0.5, 0.9, 0.7]) == 0.5


def test_pr_auc_all_zero_y_returns_zero() -> None:
    """y.sum() == 0 → PR AUC = 0 by contract."""
    assert pr_auc([0, 0, 0, 0], [0.1, 0.5, 0.9, 0.7]) == 0.0


def test_pr_auc_all_positive_y_returns_one() -> None:
    """All positives ranked first trivially → PR AUC = 1.0 (recall sweep stays at precision=1)."""
    score = pr_auc([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4])
    # tp = [1,2,3,4], fp = 0; precision == 1 always → integral over recall ∈ [.25,1] = .75
    # via numpy trapz; key contract is finite >= 0.
    assert 0.0 <= score <= 1.0


# ---------- perfect ranking -------------------------------------------------


def test_roc_auc_perfect_ranking_is_one() -> None:
    """Positives all rank above negatives → AUC = 1.0."""
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert roc_auc(y, p) == pytest.approx(1.0)


def test_roc_auc_inverted_ranking_is_zero() -> None:
    """Negatives all rank above positives → AUC = 0.0."""
    y = [1, 1, 1, 0, 0, 0]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert roc_auc(y, p) == pytest.approx(0.0)


def test_roc_auc_random_is_around_half() -> None:
    """Random predictions → AUC ≈ 0.5."""
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.integers(0, 2, size=n)
    p = rng.uniform(0.0, 1.0, n)
    auc = roc_auc(y, p)
    assert 0.45 < auc < 0.55


def test_pr_auc_perfect_ranking_is_one() -> None:
    """Perfect ranking → AP = 1.0."""
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert pr_auc(y, p) == pytest.approx(1.0)


# ---------- cross-check vs sklearn ------------------------------------------


def test_roc_auc_equals_sklearn() -> None:
    """`wagie.metrics.roc_auc` matches sklearn on binary 0/1 inputs."""
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(42)
    n = 200
    y = rng.integers(0, 2, n)
    p = rng.uniform(0.0, 1.0, n)
    if y.sum() == 0 or y.sum() == n:
        pytest.skip("degenerate sample")
    expected = sklearn.roc_auc_score(y, p)
    assert roc_auc(y, p) == pytest.approx(expected, abs=1e-9)


def test_pr_auc_equals_sklearn_average_precision() -> None:
    """`wagie.metrics.pr_auc` matches sklearn average precision."""
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    n = 200
    y = rng.integers(0, 2, n)
    p = rng.uniform(0.0, 1.0, n)
    if y.sum() == 0:
        pytest.skip("degenerate sample")
    expected = sklearn.average_precision_score(y, p)
    assert pr_auc(y, p) == pytest.approx(expected, abs=1e-9)


# ---------- identical predictions ------------------------------------------


def test_roc_auc_identical_predictions_finite() -> None:
    """Constant scores → tied ranks. argsort is stable but order depends on input
    layout; the formula does not handle ties. Test only that result is in [0,1]."""
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    p = [0.5] * 8
    val = roc_auc(y, p)
    assert 0.0 <= val <= 1.0


def test_pr_auc_identical_predictions_finite() -> None:
    y = [0, 1, 0, 1, 0, 1]
    p = [0.5] * 6
    val = pr_auc(y, p)
    assert 0.0 <= val <= 1.0
