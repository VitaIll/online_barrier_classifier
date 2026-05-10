"""T7 + T7-meta: cheat-feature canary tests.

T7 (no-leakage canary): Build the canonical pipeline twice on a parquet that
ALSO carries a `cheat` column = log(next_minute_high / current_close). The
canonical wagie path does NOT plumb the `cheat` column anywhere (it is not in
REQUIRED_COLS in `wagie.io.sources.ParquetReplaySource`, not in any default
feature catalog). So the Brier score with-vs-without the cheat parquet should
be near-identical — leakage = 0 by construction. Asserts |delta| < 0.005.

T7-meta (canary IS sensitive): Build the same pipeline a third time, but
inject a `CheatInjectorStage` that pulls the cheat value out of the bar's
underlying minute-bars and exposes it as a feature, then point ARF at
`["cheat"]` as its only selected feature. Now the cheat is fully in the loop.
Brier should COLLAPSE to near-zero on the synthetic data (the cheat is a
near-perfect linear predictor of the next-bar excursion label). Asserts
delta > 0.05 — proving the canary actually fires when leakage is real.

Both tests share a single inner harness: build pipeline, run engine, recover
matured (z_prev, y_prev) pairs from a copy of the LabelBuffer driven over the
audited p_online sequence, compute Brier. We do this WITHOUT touching the
production engine code path.

This is the test that REVOKED rounds R-015/017/018/020/021/025/030/031 should
have caught. The audit found this canary was fictional — defined in conftest
but consumed by zero tests. This file fixes that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl
import pytest
from freezegun import freeze_time

from wagie.config import WagieConfig
from wagie.core.numeric import Probability
from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind
from wagie.engine import Engine, EngineResult
from wagie.features import (
    BaseBarFeatures,
    FeatureBuilder,
    RegimeCuts,
    RegimeFeature,
)
from wagie.features.catalog import default_streaming_features
from wagie.io.brokers import SimBroker
from wagie.io.clock import TestClock
from wagie.io.sources import ParquetReplaySource
from wagie.metrics.calibration import brier_score
from wagie.pipeline import (
    LabelBuffer,
    MondrianACICalibrator,
    OnlineARFCorrector,
    Pipeline,
)
from wagie.risk import RiskEngine
from wagie.strategy import build_strategy


# ----------------------------------------------------------------------------
# Cheat injector — a Stage that exposes a leakage-leaking feature
# ----------------------------------------------------------------------------


class CheatInjectorStage:
    """Test-only Stage. Reads `cheat` directly from the parquet (sidecar dict
    keyed by bar.ts_init.ms) and adds it to obs.features.

    Only used in T7-meta to PROVE the canary fires when leakage exists.
    """

    name: str = "cheat_injector"
    kind: StageKind = StageKind.FEATURE

    def __init__(self, cheat_by_ms: dict[int, float]):
        self._cheat_by_ms = dict(cheat_by_ms)
        self._n_seen = 0

    def transform(self, obs: Observation) -> Observation:
        try:
            ts_ms = int(obs.bar.ts_init.ms)
        except Exception:
            return obs
        v = self._cheat_by_ms.get(ts_ms, float("nan"))
        self._n_seen += 1
        return obs.with_features(cheat=float(v))

    def update(self, obs: Observation, label: Optional[int] = None) -> None:
        return None

    def state_dict(self) -> dict:
        return {"n_seen": self._n_seen}

    def load_state_dict(self, state: dict) -> None:
        self._n_seen = int(state.get("n_seen", 0))

    def state_hash(self) -> bytes:
        import hashlib
        h = hashlib.sha256()
        h.update(b"CheatInjectorStage|")
        h.update(repr(self._n_seen).encode())
        return h.digest()

    def reset(self) -> None:
        self._n_seen = 0


# ----------------------------------------------------------------------------
# Harness — build + run + collect (p_online, y_matured) pairs
# ----------------------------------------------------------------------------


@dataclass
class CanaryRun:
    n_decisions: int
    n_pairs: int
    brier: float
    p_pred: list[float]
    y_true: list[int]


def _base_cfg(parquet_path: str) -> WagieConfig:
    return WagieConfig.model_validate({
        "data": {"parquet_path": str(parquet_path), "m_minutes": 20},
        "model": {
            "catboost_path": None,
            "arf": {"n_models": 5, "lambda_value": 6.0, "seed": 42},
            "aci": {"alphas": [0.05, 0.10, 0.20], "gamma": 0.01,
                    "n_regimes": 3, "q_init": 0.5},
        },
        "strategy": {"kind": "pure_conformal", "alpha": 0.10},
        "broker": {"inventory_cap": 5},
        "runtime": {"warmup_samples": 10, "capture_audit": True},
    })


def _build_pipeline(
    cfg: WagieConfig,
    *,
    inject_cheat: bool,
    cheat_by_ms: Optional[dict[int, float]] = None,
) -> tuple[Pipeline, LabelBuffer]:
    """Construct the canonical pipeline. If `inject_cheat`, splice in a
    CheatInjectorStage AND restrict ARF to `["cheat"]` so the leakage flows
    end-to-end."""
    base_bar = BaseBarFeatures()
    fb = FeatureBuilder(default_streaming_features())
    cuts = RegimeCuts(
        feature="parkinson_var_rolling_mean_24",
        edges=(1e-6, 1e-5),
        labels=("low", "med", "high"),
    )
    regime = RegimeFeature(cuts)

    stages: list = [base_bar, fb, regime]

    if inject_cheat:
        if cheat_by_ms is None:
            raise ValueError("inject_cheat=True requires cheat_by_ms")
        stages.append(CheatInjectorStage(cheat_by_ms))
        # Force ARF to use ONLY the cheat — guarantees leakage is realized.
        selected = ["cheat"]
    else:
        selected = None

    arf = OnlineARFCorrector(
        n_models=cfg.model.arf.n_models,
        max_features=cfg.model.arf.max_features,
        lambda_value=cfg.model.arf.lambda_value,
        seed=cfg.model.arf.seed,
        selected_features=selected,
    )
    stages.append(arf)

    aci = MondrianACICalibrator(
        alphas=tuple(cfg.model.aci.alphas),
        gamma=cfg.model.aci.gamma,
        n_regimes=cfg.model.aci.n_regimes,
        q_init=cfg.model.aci.q_init,
        q_init_by_regime=cfg.model.aci.q_init_by_regime,
    )
    stages.append(aci)

    label_buf = LabelBuffer(alpha_label=cfg.broker.label_alpha)
    stages.append(label_buf)

    from wagie.core.numeric import LogReturn
    from wagie.core.time import Duration as _Dur
    strat = build_strategy(
        cfg.strategy.kind,
        alpha=cfg.strategy.alpha,
        take_profit=LogReturn(cfg.broker.take_profit_log),
        stop_loss=LogReturn(cfg.broker.stop_loss_log),
        expiry=_Dur.from_minutes(cfg.broker.expiry_minutes),
    )
    stages.append(strat)

    return Pipeline(stages), label_buf


def _collect_pairs(audit: list[dict], y_pairs_by_ts: dict[int, int]) -> tuple[list[float], list[int]]:
    """Align audited (ts_ns, p_online) with matured (ts_ns, y_label) pairs.

    The LabelBuffer matures a record from tick t when tick t+1 arrives.
    We key both audit and labels by the audit's ts_ns (which is bar.close_time.ns)
    and the label record's predict-time ts (also close_time.ns from the obs.bar
    inside _Record).
    """
    p: list[float] = []
    y: list[int] = []
    for row in audit:
        ts_ns = int(row["ts_ns"])
        if ts_ns not in y_pairs_by_ts:
            continue
        po = row.get("p_online")
        if po is None:
            continue
        p.append(float(po))
        y.append(int(y_pairs_by_ts[ts_ns]))
    return p, y


def _run_canary(
    cfg: WagieConfig,
    *,
    inject_cheat: bool,
    cheat_by_ms: Optional[dict[int, float]] = None,
) -> CanaryRun:
    """Build pipeline, run engine, materialize (p_pred, y_true) pairs, return Brier."""
    pipeline, label_buf = _build_pipeline(
        cfg, inject_cheat=inject_cheat, cheat_by_ms=cheat_by_ms,
    )
    source = ParquetReplaySource(
        parquet_path=cfg.data.parquet_path,
        m_minutes=cfg.data.m_minutes,
    )
    broker = SimBroker(
        m_minutes=cfg.data.m_minutes,
        cost=cfg.broker.cost_bps * 1e-4,
        execution_latency_minutes=cfg.broker.execution_latency_minutes,
        inventory_cap=cfg.broker.inventory_cap,
    )

    # Drive the engine with a side-channel that captures (ts_ns, y_label) every
    # time the LabelBuffer matures a record. We monkeypatch maybe_emit to peek.
    matured: dict[int, int] = {}
    orig_maybe_emit = label_buf.maybe_emit

    def _peek_maybe_emit(next_bar):
        # Inspect the front record BEFORE it's popped to capture its ts_close
        ts_ns_predict = (
            int(label_buf._records[0].ts_close_ns)
            if label_buf._records
            else None
        )
        out = orig_maybe_emit(next_bar)
        if out is not None and ts_ns_predict is not None:
            _, y = out
            matured[ts_ns_predict] = int(y)
        return out

    label_buf.maybe_emit = _peek_maybe_emit  # type: ignore[assignment]

    engine = Engine(
        source=source,
        pipeline=pipeline,
        broker=broker,
        clock=TestClock(0),
        label_buffer=label_buf,
        risk_engine=RiskEngine.default(),
        warmup_samples=cfg.runtime.warmup_samples,
        capture_audit=True,
    )
    result: EngineResult = engine.run()

    p, y = _collect_pairs(result.audit, matured)
    brier = brier_score(y, p) if p else float("nan")
    return CanaryRun(
        n_decisions=result.n_decisions,
        n_pairs=len(p),
        brier=brier,
        p_pred=p,
        y_true=y,
    )


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_T7_canary_no_leakage_when_cheat_column_present_but_not_in_features(
    synthetic_minute_parquet, synthetic_minute_parquet_with_cheat,
):
    """T7: presence of an extra `cheat` column in the parquet must NOT change
    Brier. The canonical wagie path doesn't plumb non-REQUIRED_COLS through to
    features, so |brier_with_cheat - brier_without| < 0.005.

    This is the no-leakage assertion.
    """
    cfg_clean = _base_cfg(str(synthetic_minute_parquet))
    cfg_dirty = _base_cfg(str(synthetic_minute_parquet_with_cheat))

    run_clean = _run_canary(cfg_clean, inject_cheat=False)
    run_dirty = _run_canary(cfg_dirty, inject_cheat=False)

    # Both runs should have produced *some* matured pairs.
    assert run_clean.n_pairs >= 5, (
        f"canary needs matured labels; got {run_clean.n_pairs} clean pairs"
    )
    assert run_dirty.n_pairs >= 5, (
        f"canary needs matured labels; got {run_dirty.n_pairs} dirty pairs"
    )

    delta = abs(run_dirty.brier - run_clean.brier)
    assert delta < 0.005, (
        f"T7 FAILED: presence of `cheat` column shifted Brier by {delta:.4g} "
        f"(clean={run_clean.brier:.4g}, dirty={run_dirty.brier:.4g}). "
        "If you see this, the engine has started consuming non-REQUIRED_COLS "
        "from the parquet — investigate ParquetReplaySource and feature catalog."
    )


def _build_decision_bar_aligned_cheat(parquet_path: Path, m_minutes: int,
                                       alpha_label: float) -> dict[int, float]:
    """Compute, per decision-bar close, the LABEL VALUE itself (log next
    decision-bar max-high / current decision-bar close), then map back to the
    OPEN_TIME ms of the FIRST minute of that decision bar (which is what
    obs.bar.ts_init.ms reports). This is the maximally-leaking cheat: it IS
    the LabelBuffer's `log_excursion` value the engine will compare against
    `alpha_label` to mature y.
    """
    df = pl.read_parquet(str(parquet_path)).sort("open_time")
    open_time = df["open_time"].to_list()
    high = df["high"].to_list()
    close = df["close"].to_list()
    n = len(close)

    out: dict[int, float] = {}
    M = int(m_minutes)
    # Decision bar k spans minutes [k*M : (k+1)*M].
    # ts_init of decision bar k = open_time of minute k*M (per ParquetReplaySource:
    # `first.ts_init` is used as the decision bar's ts_init).
    # close of decision bar k = close of minute (k+1)*M - 1.
    # The next decision bar k+1 spans [(k+1)*M : (k+2)*M], so its high =
    # max(high[(k+1)*M : (k+2)*M]).
    for k in range(0, n // M):
        first_idx = k * M
        last_idx = (k + 1) * M - 1
        next_lo = (k + 1) * M
        next_hi = (k + 2) * M
        if next_hi > n:
            break
        c = float(close[last_idx])
        nh = max(float(h) for h in high[next_lo:next_hi])
        if c <= 0 or nh <= 0:
            continue
        cheat_val = math.log(nh / c)
        out[int(open_time[first_idx])] = cheat_val
    return out


def test_T7_meta_canary_fires_when_cheat_is_injected_into_features(
    synthetic_minute_parquet,
):
    """T7-meta: PROVE the canary is sensitive. Compute the LABEL itself per
    decision-bar (log next-bar max-high / current close), inject it as a
    feature, point ARF at `["cheat"]` only. ARF should learn this perfectly
    within a few hundred bars: cheat > alpha_label ⇔ y == 1.

    Asserts brier_baseline - brier_cheating > 0.05. If this fails, the canary
    is INSENSITIVE and T7 above proves nothing.
    """
    cfg = _base_cfg(str(synthetic_minute_parquet))

    # The cheat is the label-value itself, keyed by decision-bar ts_init.ms.
    cheat_by_ms = _build_decision_bar_aligned_cheat(
        synthetic_minute_parquet,
        m_minutes=cfg.data.m_minutes,
        alpha_label=cfg.broker.label_alpha,
    )

    run_baseline = _run_canary(cfg, inject_cheat=False)
    run_cheating = _run_canary(cfg, inject_cheat=True, cheat_by_ms=cheat_by_ms)

    assert run_baseline.n_pairs >= 5, (
        f"canary needs matured labels; got {run_baseline.n_pairs} baseline pairs"
    )
    assert run_cheating.n_pairs >= 5, (
        f"canary needs matured labels; got {run_cheating.n_pairs} cheating pairs"
    )

    improvement = run_baseline.brier - run_cheating.brier
    assert improvement > 0.05, (
        f"T7-meta FAILED: cheat injection improved Brier by only "
        f"{improvement:.4g} (baseline={run_baseline.brier:.4g}, "
        f"cheating={run_cheating.brier:.4g}). "
        "The canary cannot detect leakage of this magnitude — it is INSENSITIVE "
        "and T7 above gives a false sense of security. Investigate ARF / "
        "selected_features wiring."
    )


@freeze_time("2026-05-10 12:00:00")
def test_T7_audit_captures_p_online_at_every_decision(synthetic_minute_parquet):
    """Pre-condition for T7: the engine MUST record p_online in audit when
    capture_audit=True. If this regresses, the canary silently degrades to
    'no labels collected' and would skip its real assertion.

    NOTE: `@freeze_time` here is documentary — it pins wall-clock so that any
    accidental call to `time.time()` (run_id generation, MLflow timestamps,
    etc.) is reproducible if this test ever starts depending on them. The
    engine itself uses `TestClock` which is independent of wall-clock. If you
    add a real-clock dependency anywhere in the canary path, freeze_time will
    catch the resulting flake.
    """
    cfg = _base_cfg(str(synthetic_minute_parquet))
    run = _run_canary(cfg, inject_cheat=False)
    # We must have collected SOME pairs; otherwise the audit/label plumbing
    # is broken and T7's "delta < 0.005" would pass vacuously.
    assert run.n_pairs >= 10, (
        f"audit/label plumbing is broken: only {run.n_pairs} (p_online, "
        f"y_matured) pairs collected. T7 cannot fire."
    )
    # Predictions should not all be exactly 0.5 (the ARF uninitialised default).
    n_nonneutral = sum(1 for p in run.p_pred if abs(p - 0.5) > 1e-9)
    assert n_nonneutral >= 1, (
        f"all {run.n_pairs} p_online predictions are exactly 0.5 — ARF "
        "never updated; T7 would be vacuous."
    )
