"""Smoke tests for the H-107 plot helpers in src/plotting.py.

These tests do NOT verify pixel-level visual output — that's what the
round's diagnostic-script Read-back-the-PNG pass is for. They DO verify:

- Each helper runs without exception on representative inputs.
- Axes/Figure objects are returned (composable into multi-panel figures).
- Required title / axis labels are populated.
- Argument validation (mismatched lengths, missing columns) raises
  cleanly rather than producing silent garbage.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.plotting import (
    plot_calibration_by_regime,
    plot_calibration_curve,
    plot_feature_importance,
    plot_threshold_curves,
)
from src.utils import threshold_analysis


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def yp():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.uniform(size=n) < 0.10).astype(int)
    p = np.clip(0.10 + 0.15 * (y - 0.5) + 0.05 * rng.normal(size=n), 0.001, 0.999)
    return y, p


@pytest.fixture
def yp_with_regime(yp):
    rng = np.random.default_rng(1)
    y, p = yp
    vol = rng.exponential(scale=1.0, size=len(y))
    return y, p, vol


# -----------------------------------------------------------------------------
# plot_calibration_curve
# -----------------------------------------------------------------------------

def test_calibration_curve_returns_ax_with_labels(yp):
    y, p = yp
    ax = plot_calibration_curve(y, p)
    assert ax is not None
    assert "predicted" in ax.get_xlabel().lower()
    assert "frequency" in ax.get_ylabel().lower()
    plt.close("all")


def test_calibration_curve_title_includes_ece(yp):
    y, p = yp
    ax = plot_calibration_curve(y, p, show_ece=True)
    assert "ECE=" in ax.get_title()
    plt.close("all")


def test_calibration_curve_no_ece_when_disabled(yp):
    y, p = yp
    ax = plot_calibration_curve(y, p, show_ece=False)
    assert "ECE=" not in ax.get_title()
    plt.close("all")


def test_calibration_curve_accepts_existing_ax(yp):
    """Composability check: caller passes their own ax, helper draws on it."""
    y, p = yp
    fig, ax = plt.subplots()
    returned = plot_calibration_curve(y, p, ax=ax)
    assert returned is ax
    plt.close(fig)


# -----------------------------------------------------------------------------
# plot_calibration_by_regime
# -----------------------------------------------------------------------------

def test_calibration_by_regime_default_3_regimes(yp_with_regime):
    y, p, vol = yp_with_regime
    fig = plot_calibration_by_regime(y, p, vol)
    assert len(fig.axes) == 3
    titles = [ax.get_title() for ax in fig.axes]
    for label in ("low", "med", "high"):
        assert any(label in t for t in titles), f"missing {label} panel"
    plt.close(fig)


def test_calibration_by_regime_custom_n_and_labels(yp_with_regime):
    y, p, vol = yp_with_regime
    fig = plot_calibration_by_regime(
        y, p, vol, n_regimes=4, labels=("a", "b", "c", "d"),
    )
    assert len(fig.axes) == 4
    plt.close(fig)


def test_calibration_by_regime_rejects_label_count_mismatch(yp_with_regime):
    y, p, vol = yp_with_regime
    with pytest.raises(ValueError, match="labels length"):
        plot_calibration_by_regime(y, p, vol, n_regimes=3, labels=("a", "b"))


# -----------------------------------------------------------------------------
# plot_feature_importance
# -----------------------------------------------------------------------------

def test_feature_importance_top_n_capped_to_array_length():
    imps = np.array([0.5, 0.3, 0.2])
    names = ["a", "b", "c"]
    ax = plot_feature_importance(imps, names, top_n=20)
    # We asked for 20 but only 3 features exist.
    bars = [p.get_label() for p in ax.containers[0]] if ax.containers else None
    assert len(ax.patches) == 3
    plt.close("all")


def test_feature_importance_orders_descending():
    """Top-of-axis = most important. matplotlib draws barh bottom-up so the
    last patch added is the most important; we pre-reversed in the helper.
    """
    imps = np.array([0.1, 0.9, 0.4])
    names = ["small", "biggest", "middle"]
    ax = plot_feature_importance(imps, names, top_n=3)
    # The y-axis tick labels should read [biggest, middle, small] from top down.
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    # matplotlib ytick order: bottom-to-top maps to index 0..N-1.
    # We reversed so top = biggest.
    assert yticklabels[-1] == "biggest"
    assert yticklabels[0] == "small"
    plt.close("all")


def test_feature_importance_rejects_misaligned_lengths():
    imps = np.array([0.1, 0.2, 0.3])
    names = ["a", "b"]
    with pytest.raises(ValueError, match="must align"):
        plot_feature_importance(imps, names)


def test_feature_importance_custom_title():
    imps = np.array([0.5, 0.5])
    names = ["x", "y"]
    ax = plot_feature_importance(imps, names, title="Round 003")
    assert "Round 003" in ax.get_title()
    plt.close("all")


# -----------------------------------------------------------------------------
# plot_threshold_curves
# -----------------------------------------------------------------------------

def test_threshold_curves_three_lines(yp):
    y, p = yp
    df = threshold_analysis(y, p)
    ax = plot_threshold_curves(df)
    # Trade rate, precision, recall.
    assert len(ax.get_lines()) == 3
    legend_texts = [t.get_text().lower() for t in ax.get_legend().get_texts()]
    assert any("trade" in t for t in legend_texts)
    assert any("precision" in t for t in legend_texts)
    assert any("recall" in t for t in legend_texts)
    plt.close("all")


def test_threshold_curves_rejects_missing_columns():
    df = pd.DataFrame({"threshold": [0.0, 0.5], "trade_rate": [1.0, 0.5]})
    with pytest.raises(ValueError, match="missing columns"):
        plot_threshold_curves(df)
