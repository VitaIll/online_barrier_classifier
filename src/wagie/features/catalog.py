"""Feature catalogs — assemble FeatureBuilders for common configurations.

`default_streaming_catalog()` produces a FeatureBuilder of BaseBarFeatures
plus rolling means/vars over standard windows. The catalog is
`BaseBarFeatures` followed by a list of RiverRollingFeature / LagFeature /
PolarsBatchFeature instances. The FeatureBuilder reads intermediates
produced by BaseBarFeatures (return, log_close, parkinson_var, …).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from river import stats

from wagie.features.base import (
    FeatureBuilder,
    FeatureKind,
    FutureCovariateSpec,
    PastCovariateSpec,
)
from wagie.features.base_bar import BaseBarFeatures
from wagie.features.bounded_batch import (
    BoundedBatchFeature,
    make_hurst,
    make_sample_entropy,
)
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

# DFA, Hurst, sample entropy windows.
_BOUNDED_BATCH_WINDOW = 128


def _load_importance(path: Path) -> dict:
    """Load a feature-importance dict from a JSON or simple `name<TAB>score` file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
        if isinstance(data, list):
            # list of {name, score}
            return {str(d["name"]): float(d["score"]) for d in data if "name" in d}
    except json.JSONDecodeError:
        pass
    out: dict = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return out


def _dedupe_correlated(
    feats: list,
    *,
    threshold: float = 0.95,
    n_synthetic: int = 256,
    seed: int = 0,
) -> list:
    """Greedy de-duplication: drop features whose pairwise abs(corr) on a small
    synthetic stream exceeds `threshold` against an already-kept feature.

    The synthetic stream is intentionally tiny — this is a coarse filter to
    eliminate redundant rolling stats (e.g. mean & sum on the same source +
    window), not a substitute for proper feature selection.
    """
    if not feats:
        return feats
    rng = np.random.default_rng(seed)
    sources = sorted({getattr(f, "_source_col", None) for f in feats
                      if getattr(f, "_source_col", None) is not None})
    if not sources:
        return list(feats)
    # Synthetic source streams: independent gaussians.
    src_streams = {s: rng.normal(0.0, 1.0, size=n_synthetic) for s in sources}
    # Always include some "always-on" features for calendar columns.
    src_streams["minute_sin"] = np.sin(np.linspace(0, 2 * np.pi, n_synthetic))
    src_streams["minute_cos"] = np.cos(np.linspace(0, 2 * np.pi, n_synthetic))
    src_streams["dow_sin"] = np.sin(np.linspace(0, 2 * np.pi, n_synthetic))
    src_streams["dow_cos"] = np.cos(np.linspace(0, 2 * np.pi, n_synthetic))
    src_streams["segment_id"] = np.zeros(n_synthetic, dtype=np.int64)

    # Compute each feature's stream by running update_one on a fresh copy.
    feat_outputs: dict[str, np.ndarray] = {}
    for f in feats:
        name = f.spec.name
        try:
            f.reset_state()
        except Exception:
            pass
        out = np.empty(n_synthetic, dtype=np.float64)
        for i in range(n_synthetic):
            x = {s: float(src_streams[s][i]) for s in src_streams}
            try:
                d = f.update_one(x)
                out[i] = float(d.get(name, np.nan))
            except Exception:
                out[i] = np.nan
        # Reset post-walk so the feature is fresh for real use.
        try:
            f.reset_state()
        except Exception:
            pass
        feat_outputs[name] = out

    kept: list = []
    kept_streams: list[np.ndarray] = []
    for f in feats:
        name = f.spec.name
        cur = feat_outputs[name]
        # Only consider finite portion for correlation.
        ok = np.isfinite(cur)
        if ok.sum() < 8:
            kept.append(f)
            kept_streams.append(cur)
            continue
        too_corr = False
        for prev in kept_streams:
            both = ok & np.isfinite(prev)
            if both.sum() < 8:
                continue
            a = cur[both]
            b = prev[both]
            sa = a.std()
            sb = b.std()
            if sa <= 0 or sb <= 0:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(c):
                continue
            if abs(c) > threshold:
                too_corr = True
                break
        if not too_corr:
            kept.append(f)
            kept_streams.append(cur)
    return kept


def default_streaming_features(
    *,
    importance_path: Optional[Path] = None,
    include_bounded_batch: bool = False,
    dedupe_corr_threshold: Optional[float] = None,
) -> list:
    """Comprehensive streaming feature catalog producing ~600 features across:
        - mean, var, min, max, iqr, ptp rolling stats per (source, window)
        - lags 1..10 on a few sources
        - identity passthroughs for cyclical and per-bar derived features

    Parameters
    ----------
    importance_path : Path, optional
        Path to a JSON or whitespace-separated `name score` file. When given,
        the catalog is sorted by descending importance (unknown names sink to
        the bottom).
    include_bounded_batch : bool
        When True, append bounded-batch numba kernels (Hurst, sample-entropy)
        over a 128-bar window. Off by default — these are O(W) per bar and
        much costlier than the streaming Welford stats.
    dedupe_corr_threshold : float, optional
        When provided AND `importance_path` is set, drop features whose
        pairwise abs(correlation) on a small synthetic stream exceeds this
        threshold (default behaviour: no de-dup unless asked).
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

    # Optional bounded-batch numba features (off by default — O(W) per bar).
    if include_bounded_batch:
        feats.append(make_hurst(
            name=f"hurst_{_BOUNDED_BATCH_WINDOW}",
            source_col="log_close",
            window=_BOUNDED_BATCH_WINDOW,
            lag=1,
        ))
        feats.append(make_sample_entropy(
            name=f"sample_entropy_{_BOUNDED_BATCH_WINDOW}",
            source_col="return",
            window=_BOUNDED_BATCH_WINDOW,
            lag=1,
        ))
        # DFA-alpha is currently Hurst-equivalent in this codebase (the kernel
        # in bounded_batch.py is R/S Hurst — we expose it under a second name
        # so callers can opt in for compatibility without instantiating twice).
        feats.append(make_hurst(
            name=f"dfa_alpha_{_BOUNDED_BATCH_WINDOW}",
            source_col="return",
            window=_BOUNDED_BATCH_WINDOW,
            lag=1,
        ))

    # Importance-aware sort + optional de-dup.
    if importance_path is not None:
        importance = _load_importance(Path(importance_path))
        # Rank: descending importance. Unknown names get -inf (sink to end).
        feats.sort(key=lambda f: importance.get(f.spec.name, float("-inf")),
                   reverse=True)
        if dedupe_corr_threshold is not None:
            feats = _dedupe_correlated(feats, threshold=float(dedupe_corr_threshold))

    return feats


def default_streaming_catalog(
    regime_cuts: Optional[RegimeCuts] = None,
    *,
    importance_path: Optional[Path] = None,
    include_bounded_batch: bool = False,
) -> tuple[BaseBarFeatures, FeatureBuilder, Optional[RegimeFeature]]:
    """Returns (base_bar, feature_builder, regime_feature_or_None).

    Pipeline composition order:
        base_bar | feature_builder | (regime_feature)? | <model> | ACI | strategy

    `base_bar` derives per-bar quantities (return, log_close, parkinson_var, ...).
    `feature_builder` walks the rolling/lag features; reads from base_bar's outputs.
    `regime_feature` (optional) bins parkinson_var_rolling_mean_24 into terciles.
    """
    base_bar = BaseBarFeatures()
    fb = FeatureBuilder(default_streaming_features(
        importance_path=importance_path,
        include_bounded_batch=include_bounded_batch,
    ))
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
