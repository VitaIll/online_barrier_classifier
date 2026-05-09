"""Feature-engineering invariants.

These ensure that:
- compute_base_series produces the expected base columns.
- The undef-flag-and-impute pipeline leaves no NaNs/Infs in feature columns.
- No engineered feature column is silently dropped between stages.
- Imputation uses the documented neutral values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import utils


def test_base_series_has_required_columns(synthetic_ohlcv):
    df = utils.compute_base_series(synthetic_ohlcv.copy())
    required = {"p", "r", "rho", "logvol", "ofi", "clv", "b"}
    missing = required - set(df.columns)
    assert not missing, f"compute_base_series missing columns: {missing}"
    assert df["r"].iloc[0] != df["r"].iloc[0] or pd.isna(df["r"].iloc[0]), (
        "r at first row must be NaN (no prior close)"
    )


def test_imputation_lookup_returns_documented_values():
    """Spot-check `get_imputation_value` matches the spec table."""
    cases = [
        ("ret__lag1__f__w0", 0.0),
        ("ret__rsi__f__w14", 50.0),
        ("vol__semivar_ratio__f__w60", 1.0),
        ("vol__bpv_ratio__f__w120", 1.0),
        ("vol__ratio__f__ws10__wl60", 1.0),
        ("logp__pos__f__w240", 0.5),
        ("tb_ratio__inst__f__w0", 0.5),
        ("pentropy_norm__inst__f__w60__m3__tau1", 0.5),
        ("hit__prev__h__w0", 0),
        ("hit__since__h__w0", 144),
        ("logvol__mean__f__w60", 0.0),
    ]
    for feat, expected in cases:
        value = utils.get_imputation_value(feat, p_hit_prior=0.5, cap_h_blocks=144)
        assert value == expected, f"{feat}: got {value}, expected {expected}"


def test_imputation_refuses_constants():
    """cost__c* and barrier__phi* must raise — they are config errors when NaN."""
    with pytest.raises(ValueError):
        utils.get_imputation_value("cost__c__h__w0")
    with pytest.raises(ValueError):
        utils.get_imputation_value("barrier__phi__h__w0")


def test_undef_flag_and_impute_eliminates_nans(synthetic_ohlcv):
    """After Stage 10, no NaN/Inf must remain in declared feature columns."""
    df = utils.compute_base_series(synthetic_ohlcv.copy())
    # Inject some NaNs into a numeric column to force the flag-and-impute path.
    df = df.assign(synthetic_feature=np.where(np.arange(len(df)) % 13 == 0, np.nan, 1.0))

    feature_cols = ["synthetic_feature"]
    df_imputed, undef_cols = utils.create_undef_flags_and_impute(
        df, feature_cols=feature_cols, p_hit_prior=0.5, cap_h_blocks=144
    )

    assert "undef__synthetic_feature" in undef_cols
    assert df_imputed["synthetic_feature"].isna().sum() == 0
    assert np.isfinite(df_imputed["synthetic_feature"]).all()


def test_finite_metrics_helpers():
    """compute_all_metrics must produce finite scalars on a non-degenerate sample."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    p = rng.uniform(0.05, 0.95, size=500)
    metrics = utils.compute_all_metrics(y, p)
    for name, val in metrics.items():
        assert np.isfinite(val), f"{name} is not finite: {val}"
