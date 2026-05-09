"""Feature catalogs — assemble FeatureBuilders for common configurations.

`default_streaming_catalog()` produces a FeatureBuilder of BaseBarFeatures
plus rolling means/vars over standard windows. The catalog is
`BaseBarFeatures` followed by a list of RiverRollingFeature / LagFeature /
PolarsBatchFeature instances. The FeatureBuilder reads intermediates
produced by BaseBarFeatures (return, log_close, parkinson_var, …).
"""

from __future__ import annotations

from typing import Iterable, Optional

from river import stats

from wagie.features.base import (
    FeatureBuilder,
    FeatureKind,
    FutureCovariateSpec,
    PastCovariateSpec,
)
from wagie.features.base_bar import BaseBarFeatures
from wagie.features.regime import RegimeCuts, RegimeFeature
from wagie.features.streaming import (
    IdentityFeature,
    LagFeature,
    RiverRollingFeature,
)
from wagie.features.stats_extra import make_iqr, make_ptp


DEFAULT_WINDOWS = (1, 2, 4, 8, 12, 24, 48, 96)
DEFAULT_LAGS = (1, 2, 3, 5, 10)
DEFAULT_ROLLING_BASE = (
    "return", "abs_return", "squared_return", "range_pct",
    "parkinson_var", "garman_klass_var", "log_volume",
    "log_quote_volume", "log_trades", "range_hl",
    "close_to_vwap", "imbalance", "buy_ratio",
)
DEFAULT_LAG_BASE = ("return", "abs_return", "range_hl", "log_volume", "buy_ratio", "imbalance")


def default_streaming_features() -> list:
    """Comprehensive streaming feature catalog producing ~600 features across:
        - mean, var, min, max, iqr, ptp rolling stats per (source, window)
        - lags 1..10 on a few sources
        - identity passthroughs for cyclical and per-bar derived features
    """
    feats: list = []

    # Stats: river-native (Mean / Var revertable; RollingMin/Max have built-in window)
    # + custom (iqr, ptp). Each entry: (name, factory, only_if_window>=N)
    stat_specs = [
        ("mean", stats.Mean,       1),    # always
        ("var",  stats.Var,        2),    # only if window >= 2
        ("min",  stats.RollingMin, 1),    # built-in window
        ("max",  stats.RollingMax, 1),    # built-in window
        ("iqr",  make_iqr,         4),    # only if window >= 4
        ("ptp",  make_ptp,         2),    # only if window >= 2
    ]

    for src in DEFAULT_ROLLING_BASE:
        for w in DEFAULT_WINDOWS:
            for stat_name, factory, min_w in stat_specs:
                if w < min_w:
                    continue
                feats.append(RiverRollingFeature(
                    spec=PastCovariateSpec(
                        name=f"{src}_rolling_{stat_name}_{w}",
                        kind=FeatureKind.STREAMING, lag=1, window=w,
                    ),
                    source_col=src,
                    stat_factory=factory,
                ))

    # Lags
    for src in DEFAULT_LAG_BASE:
        for lag in DEFAULT_LAGS:
            feats.append(LagFeature(
                spec=PastCovariateSpec(
                    name=f"{src}_lag_{lag}",
                    kind=FeatureKind.STREAMING, lag=lag, window=1,
                ),
                source_col=src,
            ))

    # Calendar features as FutureCovariate (knowable at decision time, lag=0).
    # BaseBarFeatures already emits minute_sin/cos and dow_sin/cos directly into
    # the dict; we add Identity passes so they're explicit in the catalog.
    for col in ("minute_sin", "minute_cos", "dow_sin", "dow_cos"):
        feats.append(IdentityFeature(
            spec=FutureCovariateSpec(name=col + "_id", kind=FeatureKind.STREAMING,
                                     lag=0, window=1),
            source_col=col,
        ))

    return feats


def default_streaming_catalog(
    regime_cuts: Optional[RegimeCuts] = None,
) -> tuple[BaseBarFeatures, FeatureBuilder, Optional[RegimeFeature]]:
    """Returns (base_bar, feature_builder, regime_feature_or_None).

    Pipeline composition order:
        base_bar | feature_builder | (regime_feature)? | <model> | ACI | strategy

    `base_bar` derives per-bar quantities (return, log_close, parkinson_var, ...).
    `feature_builder` walks the rolling/lag features; reads from base_bar's outputs.
    `regime_feature` (optional) bins parkinson_var_rolling_mean_24 into terciles.
    """
    base_bar = BaseBarFeatures()
    fb = FeatureBuilder(default_streaming_features())
    regime = RegimeFeature(regime_cuts) if regime_cuts is not None else None
    return base_bar, fb, regime


def make_default_regime_cuts(parkinson_24_values) -> RegimeCuts:
    """Convenience: fit tercile cuts on parkinson_var_rolling_mean_24 over training."""
    return RegimeCuts.from_quantiles(
        parkinson_24_values,
        feature="parkinson_var_rolling_mean_24",
        n_regimes=3,
        labels=("low", "med", "high"),
    )


__all__ = [
    "DEFAULT_WINDOWS",
    "DEFAULT_LAGS",
    "default_streaming_features",
    "default_streaming_catalog",
    "make_default_regime_cuts",
]
