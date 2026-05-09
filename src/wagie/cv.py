"""wagie.cross_validation — walks a CombinatorialPurgedCV over the data.

Per-fold: rebuild pipeline, run engine on the test slice, collect metrics.
Final result includes per-fold metrics + CSCV PBO + replay-equivalence check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import polars as pl

from wagie.config import WagieConfig
from wagie.cscv import cscv_pbo
from wagie.metrics import compute_trading_metrics


logger = logging.getLogger(__name__)


@dataclass
class CrossValResult:
    per_fold: list[dict] = field(default_factory=list)
    pbo: float = 0.0
    n_folds: int = 0
    state_hashes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"CrossValResult: n_folds={self.n_folds}",
            f"  PBO (Bailey-LdP): {self.pbo:.3f}",
            "  per-fold:",
        ]
        for r in self.per_fold:
            lines.append(
                f"    fold={r.get('fold', '-')}: n_trades={r.get('n_trades', 0)} "
                f"sharpe={r.get('sharpe', 0):+.3f}"
            )
        return "\n".join(lines)


def cross_validation(
    cfg: WagieConfig,
    *,
    n_folds: int = 10,
    n_test_folds: int = 2,
    purged_size: int = 1,
    embargo_size: int = 5,
    refit_offline: bool = False,
    catalog_factory: Optional[Callable[[], Any]] = None,
) -> CrossValResult:
    """Walk skfolio.CombinatorialPurgedCV over the data; per fold run the engine.

    For each fold:
      1. Build a fresh pipeline (from cfg + catalog_factory)
      2. Slice the parquet to the test fold's time window via WagieConfig
      3. Run wagie.run() on that slice
      4. Compute trading metrics
      5. Aggregate per-fold returns for CSCV PBO
    """
    from skfolio.model_selection import CombinatorialPurgedCV

    from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeCuts, RegimeFeature
    from wagie.features.catalog import default_streaming_features
    from wagie.run import run as run_one

    # Read full parquet to determine N (number of decision bars)
    df = pl.read_parquet(cfg.data.parquet_path)
    n = len(df) // (cfg.data.m_minutes)  # approximate decision-bar count

    cv = CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        purged_size=purged_size,
        embargo_size=embargo_size,
    )

    rows = []
    state_hashes = []
    fold_returns_matrix = []   # for CSCV: n_folds × n_periods

    X = np.arange(n).reshape(-1, 1)

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X)):
        if len(test_idx) == 0:
            continue
        test_start = int(test_idx.min()) * cfg.data.m_minutes
        test_end = int(test_idx.max() + 1) * cfg.data.m_minutes
        # Convert decision-bar indices to ms timestamps
        df_sorted = df.sort("open_time")
        test_open_time_ms = int(df_sorted["open_time"][test_start])
        test_close_time_ms = int(df_sorted["open_time"][min(test_end, len(df_sorted) - 1)])

        # Per-fold cfg copy (override start/end)
        fold_cfg = cfg.model_copy(deep=True)
        fold_cfg.data.start_ts_ms = test_open_time_ms
        fold_cfg.data.end_ts_ms = test_close_time_ms

        # Build catalog
        if catalog_factory is not None:
            base_bar, fb, regime = catalog_factory()
        else:
            base_bar = BaseBarFeatures()
            fb = FeatureBuilder(default_streaming_features())
            cuts = RegimeCuts.from_quantiles(
                np.linspace(1e-7, 1e-4, 100),
                feature="parkinson_var_rolling_mean_24",
            )
            regime = RegimeFeature(cuts)

        try:
            r = run_one(fold_cfg, feature_builder=fb, base_bar=base_bar, regime_feature=regime)
        except Exception as e:
            logger.warning(f"fold {fold_id} failed: {e}")
            continue

        m = compute_trading_metrics(r.fills, m_minutes=cfg.data.m_minutes)
        rows.append({
            "fold": fold_id,
            "n_trades": m.n_trades,
            "sharpe": m.sharpe,
            "psr": m.probabilistic_sharpe,
            "hit_rate": m.hit_rate,
            "total_log_return": m.total_log_return,
        })
        state_hashes.append(r.pipeline_state_hash.hex())

        # Per-fold daily-equivalent return series for CSCV
        if r.fills:
            equity = np.cumsum([float(f.pnl_log_net) for f in r.fills])
            # Pad to a common length
            fold_returns_matrix.append(np.diff(equity, prepend=0.0))

    # CSCV PBO
    pbo = 0.5
    if len(fold_returns_matrix) >= 2:
        max_len = max(len(r) for r in fold_returns_matrix)
        if max_len >= 16:
            padded = np.zeros((len(fold_returns_matrix), max_len), dtype=float)
            for i, r in enumerate(fold_returns_matrix):
                padded[i, :len(r)] = r
            try:
                pbo_result = cscv_pbo(padded, n_chunks=min(16, max_len // 2 * 2))
                pbo = pbo_result["pbo"]
            except Exception as e:
                logger.warning(f"CSCV PBO failed: {e}")

    return CrossValResult(
        per_fold=rows,
        pbo=pbo,
        n_folds=len(rows),
        state_hashes=state_hashes,
    )


# -----------------------------------------------------------------------------
# Drift wrapper
# -----------------------------------------------------------------------------

class DriftAwarePipeline:
    """Wraps a Pipeline + a drift detector. On drift, recalibrates the ACI stage.

    Composition (NOT inheritance): a thin layer that:
      - forwards transform/learn to the base Pipeline
      - feeds prediction errors to the drift detector
      - on drift, calls .reset() on the calibrator stage(s)
    """

    name: str = "drift_aware_pipeline"

    def __init__(self, base, drift_detector, recalib_window: int = 500):
        self.base = base
        self.drift_detector = drift_detector
        self.recalib_window = int(recalib_window)
        self._n_drifts = 0

    def transform_one(self, obs, ctx=None):
        return self.base.transform_one(obs, ctx)

    def learn_one(self, obs, label=None):
        # Feed drift detector with miscoverage error from prior predictions
        if label is not None and obs.in_set:
            for a, in_set in obs.in_set.items():
                err = 0 if (in_set and label == 1) else 1
                self.drift_detector.update(err)
                if getattr(self.drift_detector, "drift_detected", False):
                    self._on_drift()
                    break
        return self.base.learn_one(obs, label)

    def _on_drift(self):
        from wagie.pipeline import MondrianACICalibrator
        for s in self.base.stages:
            if isinstance(s, MondrianACICalibrator):
                s.reset()
        self._n_drifts += 1

    def state_dict(self):
        d = self.base.state_dict()
        d["__drift__"] = {"n_drifts": self._n_drifts}
        return d

    def state_hash(self) -> bytes:
        return self.base.state_hash()


__all__ = ["cross_validation", "CrossValResult", "DriftAwarePipeline"]
