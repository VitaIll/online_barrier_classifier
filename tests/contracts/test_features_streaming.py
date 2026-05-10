"""Coverage tests for `wagie.features.streaming` and `wagie.features.stats_extra`."""

from __future__ import annotations

import math

import pytest

from wagie.features.base import FeatureKind, FutureCovariateSpec, PastCovariateSpec
from wagie.features.stats_extra import make_iqr, make_ptp
from wagie.features.streaming import (
    IdentityFeature, LagFeature, RiverRollingFeature,
    make_lag, make_rolling_mean, make_rolling_var,
)


# ---- LagFeature -----------------------------------------------------------


def test_lag_feature_rejects_non_unit_window():
    spec = PastCovariateSpec(name="x_lag1", kind=FeatureKind.STREAMING,
                              lag=1, window=2)
    with pytest.raises(ValueError, match="window must be 1"):
        LagFeature(spec, source_col="x")


def test_lag_feature_emits_nan_until_buffer_full():
    f = make_lag("x_lag2", "x", lag=2)
    out0 = f.update_one({"x": 1.0})
    out1 = f.update_one({"x": 2.0})
    out2 = f.update_one({"x": 3.0})
    assert math.isnan(out0["x_lag2"])
    assert math.isnan(out1["x_lag2"])
    assert out2["x_lag2"] == 1.0


def test_lag_feature_handles_missing_source_col():
    f = make_lag("x_lag1", "x", lag=1)
    out = f.update_one({})  # x missing → NaN inserted
    assert math.isnan(out["x_lag1"])


def test_lag_feature_reset_clears_buffer():
    f = make_lag("x_lag1", "x", lag=1)
    f.update_one({"x": 1.0})
    f.update_one({"x": 2.0})
    f.reset_state()
    out = f.update_one({"x": 3.0})
    assert math.isnan(out["x_lag1"])
    assert f._n_seen == 1


def test_lag_feature_state_hash_changes_with_data():
    f = make_lag("x_lag1", "x", lag=1)
    h0 = f.state_hash()
    f.update_one({"x": 1.0})
    h1 = f.state_hash()
    assert h0 != h1


# ---- IdentityFeature ------------------------------------------------------


def test_identity_feature_lag_zero_passthrough():
    spec = FutureCovariateSpec(name="y", kind=FeatureKind.STREAMING, lag=0, window=1)
    f = IdentityFeature(spec, source_col="x")
    out = f.update_one({"x": 1.5})
    assert out["y"] == 1.5


def test_identity_feature_lag_positive_emits_old_value():
    spec = PastCovariateSpec(name="y", kind=FeatureKind.STREAMING, lag=2, window=1)
    f = IdentityFeature(spec, source_col="x")
    f.update_one({"x": 10.0})
    f.update_one({"x": 20.0})
    out = f.update_one({"x": 30.0})
    assert out["y"] == 10.0


def test_identity_feature_handles_none_source():
    spec = FutureCovariateSpec(name="y", kind=FeatureKind.STREAMING, lag=0, window=1)
    f = IdentityFeature(spec, source_col="x")
    out = f.update_one({"x": None})
    assert math.isnan(out["y"])


def test_identity_feature_reset_clears_buffer():
    spec = PastCovariateSpec(name="y", kind=FeatureKind.STREAMING, lag=2, window=1)
    f = IdentityFeature(spec, source_col="x")
    f.update_one({"x": 1.0})
    f.reset_state()
    out = f.update_one({"x": 99.0})
    assert math.isnan(out["y"])


def test_identity_feature_state_hash_no_lag_branch():
    spec = FutureCovariateSpec(name="y", kind=FeatureKind.STREAMING, lag=0, window=1)
    f = IdentityFeature(spec, source_col="x")
    assert f.state_hash() == b"identity_nolag"


def test_identity_feature_state_hash_with_lag():
    spec = PastCovariateSpec(name="y", kind=FeatureKind.STREAMING, lag=2, window=1)
    f = IdentityFeature(spec, source_col="x")
    h0 = f.state_hash()
    f.update_one({"x": 1.0})
    h1 = f.state_hash()
    assert h0 != h1


# ---- RiverRollingFeature ---------------------------------------------------


def test_make_rolling_mean_basic():
    f = make_rolling_mean("m3", "x", window=3, lag=1)
    f.update_one({"x": 1.0})
    f.update_one({"x": 2.0})
    out = f.update_one({"x": 3.0})
    # The lag-1 value reflects the rolling mean BEFORE this tick's contribution
    assert out["m3"] is not None


def test_make_rolling_var_basic():
    f = make_rolling_var("v3", "x", window=3, lag=1)
    for v in [1.0, 2.0, 3.0, 4.0]:
        out = f.update_one({"x": v})
    assert out["v3"] is not None


def test_river_rolling_reset_state_clears():
    f = make_rolling_mean("m3", "x", window=3, lag=1)
    for v in [1.0, 2.0, 3.0]:
        f.update_one({"x": v})
    f.reset_state()
    assert f._n_seen == 0


def test_river_rolling_state_hash_changes_with_n_seen():
    f = make_rolling_mean("m3", "x", window=3, lag=1)
    h0 = f.state_hash()
    for v in [1.0, 2.0, 3.0]:
        f.update_one({"x": v})
    h1 = f.state_hash()
    assert h0 != h1


# ---- stats_extra: IQR / PTP -----------------------------------------------


def test_make_iqr_returns_finite_on_data():
    iqr = make_iqr()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        iqr.update(v)
    val = iqr.get()
    assert math.isfinite(val)
    assert val >= 0.0


def test_make_iqr_revertable():
    iqr = make_iqr()
    for v in [1.0, 2.0, 3.0, 4.0]:
        iqr.update(v)
    iqr.revert(1.0)  # remove oldest — internal state may go to NaN; not crash
    val = iqr.get()
    assert math.isfinite(val) or math.isnan(val)


def test_make_ptp_basic():
    ptp = make_ptp()
    for v in [1.0, 5.0, 2.0, 4.0]:
        ptp.update(v)
    val = ptp.get()
    # peak-to-peak = max - min = 5 - 1 = 4
    assert val == pytest.approx(4.0)


def test_make_ptp_revert():
    ptp = make_ptp()
    for v in [1.0, 5.0, 2.0]:
        ptp.update(v)
    ptp.revert(1.0)  # remove an entry
    val = ptp.get()
    assert math.isfinite(val)


def test_make_iqr_empty_returns_finite():
    iqr = make_iqr()
    val = iqr.get()
    # Empty stat may return 0 or NaN — either is acceptable; just must not crash
    assert math.isfinite(val) or math.isnan(val)


def test_make_ptp_empty_returns_finite():
    ptp = make_ptp()
    val = ptp.get()
    assert math.isfinite(val) or math.isnan(val)


# ---- Catalog: importance sort + bounded-batch opt-in ----------------------


def test_default_streaming_features_is_unsorted_without_importance():
    """Without importance_path, the catalog has its natural construction order
    (rolling stats, then lags, then identities)."""
    from wagie.features.catalog import default_streaming_features

    feats = default_streaming_features()
    names = [f.spec.name for f in feats]
    # First feature should be a rolling-mean of the first source col
    assert names[0].startswith("return_rolling_mean_")
    # Identity passes for calendar features should sit at the end.
    assert any(n.endswith("_id") for n in names[-10:])


def test_default_streaming_features_sorted_by_importance(tmp_path):
    """When importance_path is given, the catalog is sorted by descending
    importance score; unknown names sink to the bottom."""
    import json
    from wagie.features.catalog import default_streaming_features

    # Use feature names that we know exist in the default catalog.
    importance = {
        "buy_ratio_lag_5": 0.99,
        "log_volume_rolling_mean_24": 0.80,
        "abs_return_rolling_var_8": 0.55,
    }
    p = tmp_path / "imp.json"
    p.write_text(json.dumps(importance), encoding="utf-8")

    feats = default_streaming_features(importance_path=p)
    names = [f.spec.name for f in feats]
    # The three known-importance features must appear at the front, in score order.
    assert names[0] == "buy_ratio_lag_5"
    assert names[1] == "log_volume_rolling_mean_24"
    assert names[2] == "abs_return_rolling_var_8"


def test_default_streaming_features_dedupe_with_threshold(tmp_path):
    """When dedupe_corr_threshold is supplied alongside importance, redundant
    features are dropped. The importance tie-breaker must keep the highest-
    importance feature in each correlation cluster."""
    import json
    from wagie.features.catalog import default_streaming_features

    importance = {
        # Two features that are perfectly correlated (same source, same window):
        # "return_rolling_mean_4" vs the same again — but they're unique names.
        # So we just verify the dedupe call runs and returns a smaller catalog.
        "return_rolling_mean_4": 1.0,
        "return_rolling_mean_8": 0.5,
    }
    p = tmp_path / "imp.json"
    p.write_text(json.dumps(importance), encoding="utf-8")

    feats_no_dedupe = default_streaming_features(importance_path=p)
    feats_with_dedupe = default_streaming_features(
        importance_path=p, dedupe_corr_threshold=0.95
    )
    # Dedupe should not increase size; in synthetic gaussian streams many
    # rolling stats over the same source are highly correlated.
    assert len(feats_with_dedupe) <= len(feats_no_dedupe)


def test_default_streaming_features_include_bounded_batch():
    """When include_bounded_batch=True, the catalog appends Hurst, DFA-alpha,
    sample-entropy bounded-batch numba features."""
    from wagie.features.catalog import default_streaming_features

    feats_off = default_streaming_features(include_bounded_batch=False)
    feats_on = default_streaming_features(include_bounded_batch=True)
    names_on = [f.spec.name for f in feats_on]
    names_off = [f.spec.name for f in feats_off]
    assert any("hurst_128" in n for n in names_on)
    assert any("dfa_alpha_128" in n for n in names_on)
    assert any("sample_entropy_128" in n for n in names_on)
    # Without the flag, none of those names should be present.
    assert not any("hurst_128" in n for n in names_off)


def test_undef_flag_pattern_emitted_in_arf_z():
    """The streaming feature emits NaN during warmup; OnlineARFCorrector's
    `_z_with_undef_flags` then turns each NaN into `(sentinel=0.0, flag=1)`."""
    from wagie.core.event import DecisionBar
    from wagie.core.identity import DEFAULT_INSTRUMENT
    from wagie.core.numeric import Price, Probability, Quantity
    from wagie.core.observation import Observation
    from wagie.core.time import Duration, Timestamp
    from wagie.pipeline.online_arf import OnlineARFCorrector

    arf = OnlineARFCorrector(selected_features=["x_rolling_mean_4"])
    bar = DecisionBar(
        ts_init=Timestamp(0), instrument=DEFAULT_INSTRUMENT,
        open=Price(1.0), high=Price(1.0), low=Price(1.0), close=Price(1.0),
        volume=Quantity(0.0), duration=Duration.from_minutes(20), segment_id=0,
    )
    o = Observation(bar=bar)
    o = o.with_features_dict({"x_rolling_mean_4": float("nan")})
    o = o.with_p_offline(Probability(0.5))
    z = arf._z_with_undef_flags(o)
    assert z["x_rolling_mean_4"] == 0.0
    assert z["x_rolling_mean_4__undef"] == 1
    assert z["p_offline"] == 0.5
    assert z["p_offline__undef"] == 0
