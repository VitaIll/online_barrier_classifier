"""ExperimentProtocol — the SINGLE protocol for running experiments.

There is exactly one way to run an experiment in this repo:

    from wagie.experiments import ExperimentSpec, ExperimentProtocol
    spec = ExperimentSpec.from_yaml("experiments/baseline.yaml")
    result = ExperimentProtocol().run(spec)

Or via CLI:

    wagie experiment run experiments/baseline.yaml

Stages (one tick — backtest):
    1. Resolve out_dir, write spec snapshot
    2. Build pipeline + engine via wagie.run.run()
    3. Run engine event loop → EngineResult
    4. MetricsBattery.compute → MetricsReport
    5. ChartBattery.render_all → png paths
    6. Report.render → report.md
    7. Persist metrics.json + state hash

When ``spec.cv`` is set, the protocol delegates to ``wagie.cv.cross_validation``
and aggregates per-fold metrics into the same MetricsReport / Report shape.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from wagie.charts import ChartBattery
from wagie.engine import EngineResult
from wagie.metrics import MetricsBattery
from wagie.reporting import Report
from wagie.run import run as run_backtest

from .result import ExperimentResult
from .spec import ExperimentSpec


logger = logging.getLogger(__name__)


def _make_run_id(spec: ExperimentSpec) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{ts}_{spec.name}_{spec.hash()[:8]}"


# =============================================================================
# Accept-gate (round-040 additive)
# =============================================================================

def _check_accept_gates(metrics: dict, spec: ExperimentSpec) -> list[str]:
    """Evaluate the spec's pre-registration gates against ``metrics``.

    Returns a list of human-readable failure reasons. Empty list = PASS.
    Gates evaluated:
      1. Sample size: ``trading.n_trades`` ≥ ``spec.min_n_trades``.
      2. Predicted-effect floor: ``trading.sharpe`` ≥
         ``spec.predicted_effect_min`` (when set). For backtest mode
         additionally requires the lower CI bound to clear zero when a
         CI was computed.
      3. Seed-band gap: when the metrics carry a ``sharpe_seed_band``
         (CV mode), the median fold Sharpe must clear the band's lo by
         at least ``spec.min_effect_vs_seed_band``.
    """
    reasons: list[str] = []
    trading = metrics.get("trading") or {}
    n_trades = int(trading.get("n_trades", 0) or 0)
    if n_trades < int(spec.min_n_trades):
        reasons.append(
            f"min_n_trades: n_trades={n_trades} < required={spec.min_n_trades}"
        )

    sharpe = float(trading.get("sharpe", 0.0) or 0.0)
    if spec.predicted_effect_min is not None:
        if sharpe < float(spec.predicted_effect_min):
            reasons.append(
                f"predicted_effect_min: sharpe={sharpe:+.4f} < "
                f"required={spec.predicted_effect_min:+.4f}"
            )
        # Also enforce CI lower bound > 0 when a Sharpe CI is present.
        sh_ci = metrics.get("sharpe_ci")
        if sh_ci and isinstance(sh_ci, (list, tuple)) and len(sh_ci) == 2:
            if float(sh_ci[0]) <= 0.0:
                reasons.append(
                    f"sharpe_ci_lo<=0: ci_lo={float(sh_ci[0]):+.4f} not > 0 "
                    "(CI contains zero — point estimate is not "
                    "statistically distinguishable from null)"
                )

    band = metrics.get("sharpe_seed_band")
    if band and isinstance(band, dict):
        med = float(band.get("med", 0.0))
        lo = float(band.get("lo", 0.0))
        gap = med - lo
        if gap < float(spec.min_effect_vs_seed_band):
            reasons.append(
                f"min_effect_vs_seed_band: med-lo gap={gap:+.4f} < "
                f"required={spec.min_effect_vs_seed_band:+.4f} "
                f"(seed-band median {med:+.4f}, lo {lo:+.4f})"
            )

    return reasons


def _write_blocked_md(
    out_dir: Path, *, run_id: str, spec: ExperimentSpec,
    reasons: list[str], metrics: dict,
) -> Path:
    """Write ``BLOCKED.md`` instead of ``report.md`` when gates fail."""
    trading = metrics.get("trading") or {}
    body = [
        f"# BLOCKED — accept-gate failure",
        "",
        f"- run_id: `{run_id}`",
        f"- hypothesis_id: `{spec.hypothesis_id}`",
        f"- spec hash: `{spec.hash()}`",
        f"- spec name: `{spec.name}`",
        "",
        "## Pre-registration",
        f"- predicted_effect_min: `{spec.predicted_effect_min}`",
        f"- min_n_trades: `{spec.min_n_trades}`",
        f"- min_effect_vs_seed_band: `{spec.min_effect_vs_seed_band}`",
        f"- n_trials_for_dsr: `{spec.n_trials_for_dsr}`",
        "",
        "## Observed",
        f"- n_trades: `{trading.get('n_trades', 0)}`",
        f"- sharpe (point): `"
        f"{format(trading['sharpe'], '+.4f') if isinstance(trading.get('sharpe'), (int, float)) else 'n/a'}`",
        f"- sharpe_ci: `{metrics.get('sharpe_ci')}`",
        f"- brier: `{metrics.get('brier')}`",
        f"- ece: `{metrics.get('ece')}`",
        "",
        "## Failed gates",
    ]
    for r in reasons:
        body.append(f"- {r}")
    body.append("")
    body.append("This run did NOT produce `report.md`. Re-run with adjusted "
                "spec (or accept this rejection) before proceeding.")
    body.append("")
    p = out_dir / "BLOCKED.md"
    p.write_text("\n".join(body), encoding="utf-8")
    return p


@dataclass
class ExperimentProtocol:
    """The ONE protocol. All experiments flow through here.

    ``metrics_battery``, ``chart_battery``, ``report`` are injectable for
    tests and bespoke users; defaults are the canonical shipped batteries.
    """

    metrics_battery: MetricsBattery = None
    chart_battery: ChartBattery = None
    report: Report = None

    def __post_init__(self):
        if self.metrics_battery is None:
            self.metrics_battery = MetricsBattery()
        if self.chart_battery is None:
            self.chart_battery = ChartBattery()
        if self.report is None:
            self.report = Report()

    # -----------------------------------------------------------------
    # Public entry point — dispatches on spec
    # -----------------------------------------------------------------

    def run(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: Optional[Path] = None,
    ) -> ExperimentResult:
        if spec.cv is not None and spec.cv.enabled:
            return self._run_cv(spec, spec_path=spec_path)
        return self._run_backtest(spec, spec_path=spec_path)

    # -----------------------------------------------------------------
    # Backtest mode
    # -----------------------------------------------------------------

    def _run_backtest(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: Optional[Path] = None,
    ) -> ExperimentResult:
        run_id = _make_run_id(spec)
        out_dir = Path(spec.artifacts.out_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.yaml").write_text(spec.to_yaml(), encoding="utf-8")
        logger.info(f"experiment[backtest] run_id={run_id} out_dir={out_dir}")

        feature_builder, base_bar, regime = _build_features(spec)
        self.metrics_battery.m_minutes = spec.wagie.data.m_minutes
        # Plumb pre-registration n_trials → DSR auto-invocation.
        self.metrics_battery.n_trials = int(spec.n_trials_for_dsr)
        engine_result: EngineResult = run_backtest(
            spec.wagie,
            feature_builder=feature_builder,
            base_bar=base_bar,
            regime_feature=regime,
        )

        metrics_report = self.metrics_battery.compute(engine_result)
        metrics_dict = metrics_report.to_dict()
        (out_dir / "metrics.json").write_text(metrics_report.to_json(), encoding="utf-8")

        chart_paths: dict[str, Path] = {}
        if spec.charts.enable:
            self.chart_battery.n_calibration_bins = spec.charts.n_calibration_bins
            chart_paths = self.chart_battery.render_all(
                engine_result, out_dir / "charts",
            )

        # Accept-gate (round-040 additive): BEFORE writing report.md, evaluate
        # pre-registration gates. If any fail, write BLOCKED.md and refuse to
        # publish a report.
        blocked_reasons = _check_accept_gates(metrics_dict, spec)
        accepted = not blocked_reasons
        report_path: Optional[Path] = None
        blocked_path: Optional[Path] = None
        if not accepted:
            blocked_path = _write_blocked_md(
                out_dir, run_id=run_id, spec=spec,
                reasons=blocked_reasons, metrics=metrics_dict,
            )
            metrics_dict["accepted"] = False
            metrics_dict["blocked_reasons"] = list(blocked_reasons)
            (out_dir / "metrics.json").write_text(
                json.dumps(metrics_dict, indent=2, default=str),
                encoding="utf-8",
            )
            logger.warning(
                f"experiment[backtest] BLOCKED run_id={run_id} "
                f"reasons={blocked_reasons}"
            )
        elif spec.report.enable:
            self.report.title = spec.report.title or f"experiment: {spec.name}"
            report_path = self.report.render(
                spec_dict=spec.model_dump(mode="json"),
                metrics=metrics_dict,
                chart_paths=chart_paths,
                out_path=out_dir / "report.md",
                run_id=run_id,
            )

        if spec.artifacts.save_state:
            state_dir = out_dir / "state"
            state_dir.mkdir(exist_ok=True)
            (state_dir / "pipeline_state_hash.txt").write_text(
                engine_result.pipeline_state_hash.hex(), encoding="utf-8",
            )

        result = ExperimentResult(
            run_id=run_id, out_dir=out_dir,
            spec_path=Path(spec_path) if spec_path else out_dir / "spec.yaml",
            engine_result=engine_result, metrics=metrics_dict,
            chart_paths=chart_paths, report_path=report_path,
            spec_hash=spec.hash(),
            accepted=accepted, blocked_reasons=blocked_reasons,
            blocked_path=blocked_path,
        )
        logger.info(result.headline)
        return result

    # -----------------------------------------------------------------
    # CV mode
    # -----------------------------------------------------------------

    def _run_cv(
        self,
        spec: ExperimentSpec,
        *,
        spec_path: Optional[Path] = None,
    ) -> ExperimentResult:
        from wagie.cv import cross_validation

        run_id = _make_run_id(spec)
        out_dir = Path(spec.artifacts.out_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.yaml").write_text(spec.to_yaml(), encoding="utf-8")
        logger.info(f"experiment[cv] run_id={run_id} out_dir={out_dir}")

        cv = spec.cv
        cv_result = cross_validation(
            spec.wagie,
            n_folds=cv.n_folds,
            n_test_folds=cv.n_test_folds,
            purged_size=cv.purged_size,
            embargo_size=cv.embargo_size,
        )

        metrics_dict = _cv_to_metrics_dict(cv_result, m_minutes=spec.wagie.data.m_minutes)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics_dict, indent=2, default=str), encoding="utf-8",
        )

        # Charts: CV gets its own minimal panel (per-fold sharpe bars + PBO marker)
        chart_paths: dict[str, Path] = {}
        if spec.charts.enable:
            chart_paths = self._render_cv_charts(cv_result, out_dir / "charts")

        # Accept-gate (round-040 additive) — same semantics as backtest mode.
        blocked_reasons = _check_accept_gates(metrics_dict, spec)
        accepted = not blocked_reasons
        report_path: Optional[Path] = None
        blocked_path: Optional[Path] = None
        if not accepted:
            blocked_path = _write_blocked_md(
                out_dir, run_id=run_id, spec=spec,
                reasons=blocked_reasons, metrics=metrics_dict,
            )
            metrics_dict["accepted"] = False
            metrics_dict["blocked_reasons"] = list(blocked_reasons)
            (out_dir / "metrics.json").write_text(
                json.dumps(metrics_dict, indent=2, default=str),
                encoding="utf-8",
            )
            logger.warning(
                f"experiment[cv] BLOCKED run_id={run_id} "
                f"reasons={blocked_reasons}"
            )
        elif spec.report.enable:
            self.report.title = spec.report.title or f"cv: {spec.name}"
            report_path = self.report.render(
                spec_dict=spec.model_dump(mode="json"),
                metrics=metrics_dict, chart_paths=chart_paths,
                out_path=out_dir / "report.md", run_id=run_id,
            )

        # CV doesn't have a single EngineResult; build a stub so ExperimentResult
        # remains uniform.
        from wagie.engine import EngineResult as _ER
        from wagie.io.brokers import BrokerLedger
        from wagie.core.portfolio import Portfolio
        stub_engine = _ER(
            ledger=BrokerLedger(fills=[], n_open_at_finalize=0, config={}),
            n_decisions=int(sum(r.get("n_trades", 0) for r in cv_result.per_fold)),
            n_filled=0, n_skipped_warmup=0,
            n_actions_approved=0, n_actions_rejected=0,
            pipeline_state_hash=b"\x00" * 32,
            final_portfolio=Portfolio(),
        )
        return ExperimentResult(
            run_id=run_id, out_dir=out_dir,
            spec_path=Path(spec_path) if spec_path else out_dir / "spec.yaml",
            engine_result=stub_engine, metrics=metrics_dict,
            chart_paths=chart_paths, report_path=report_path,
            spec_hash=spec.hash(),
            accepted=accepted, blocked_reasons=blocked_reasons,
            blocked_path=blocked_path,
        )

    def _render_cv_charts(self, cv_result, out_dir: Path) -> dict[str, Path]:
        import matplotlib.pyplot as plt
        from wagie.charts.theme import PALETTE, apply_theme, figsize

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out: dict[str, Path] = {}

        # Per-fold Sharpe bar chart
        apply_theme(plt)
        fig, ax = plt.subplots(figsize=figsize("wide"))
        if cv_result.per_fold:
            xs = [r["fold"] for r in cv_result.per_fold]
            sharpes = [r.get("sharpe", 0.0) for r in cv_result.per_fold]
            ax.bar(xs, sharpes, color=PALETTE["primary"], alpha=0.85)
            ax.axhline(0, color=PALETTE["muted"], linewidth=0.6)
            ax.set_xlabel("fold")
            ax.set_ylabel("Sharpe (annualized)")
            ax.set_title(f"per-fold Sharpe (PBO={cv_result.pbo:.2f})")
        else:
            ax.text(0.5, 0.5, "no folds", ha="center", va="center")
        path = out_dir / "01_per_fold_sharpe.png"
        fig.savefig(path)
        plt.close(fig)
        out["per_fold_sharpe"] = path
        return out


def _cv_to_metrics_dict(cv_result, *, m_minutes: int) -> dict:
    """Aggregate CV per-fold rows into the same shape as MetricsReport.

    Round-040 fix: previously hard-coded ``brier=0.0, ece=0.0`` even when
    folds contained real fills — that was the "CV silently writes zeros"
    audit finding. Now we:
      - Compute the cross-fold Sharpe **band** (5/50/95-pct of per-fold
        Sharpe) as a seed-band proxy.
      - Stationary-block bootstrap a single Sharpe CI on the concatenated
        per-fold PnL series (real number, not a mean of point estimates).
      - Mark Brier/ECE as ``None`` so downstream code cannot mistake the
        CV mode for one that actually produced calibration metrics. The
        engine in this repo doesn't capture (y_true, p_pred) pairs at the
        fill-level, so any CV calibration number would be a fabrication.
    """
    import numpy as np

    from wagie.metrics.bootstrap import bootstrap_sharpe

    folds = cv_result.per_fold or []
    if not folds:
        agg_sharpe = 0.0
        agg_n_trades = 0
        sharpe_seed_band = None
        sharpe_ci = None
    else:
        n_folds = max(len(folds), 1)
        agg_sharpe = sum(r.get("sharpe", 0.0) for r in folds) / n_folds
        agg_n_trades = sum(r.get("n_trades", 0) for r in folds)
        per_fold_sharpe = np.array([r.get("sharpe", 0.0) for r in folds],
                                   dtype=float)
        if len(per_fold_sharpe) >= 1:
            sharpe_seed_band = {
                "lo": float(np.quantile(per_fold_sharpe, 0.05)),
                "med": float(np.quantile(per_fold_sharpe, 0.50)),
                "hi": float(np.quantile(per_fold_sharpe, 0.95)),
                "n_folds": int(len(per_fold_sharpe)),
            }
        else:
            sharpe_seed_band = None

        # Stationary-block bootstrap on the concatenated per-fold-PnL proxy
        # (each per-fold row may carry total_log_return; treat per-fold totals
        # as a 1D series for a coarse CI). Folds < 2 → no CI.
        per_fold_returns = np.array(
            [r.get("total_log_return", 0.0) for r in folds], dtype=float,
        )
        if len(per_fold_returns) >= 2 and per_fold_returns.std() > 0:
            try:
                out = bootstrap_sharpe(per_fold_returns,
                                       n_resamples=2000, seed=42)
                sharpe_ci = (float(out["ci_lo"]), float(out["ci_hi"]))
            except (ValueError, RuntimeError):
                sharpe_ci = None
        else:
            sharpe_ci = None

    return {
        "mode": "cv",
        "n_folds": cv_result.n_folds,
        "pbo": cv_result.pbo,
        "per_fold": folds,
        "trading": {
            "n_trades": agg_n_trades,
            "sharpe": agg_sharpe,
            # Honest defaults — CV mode does NOT compute these per-trade fields.
            "n_tp": 0, "n_sl": 0, "n_timeout": 0,
            "hit_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_log_return": 0.0, "total_pct_return": 0.0,
            "probabilistic_sharpe": 0.0, "sortino": 0.0,
            "max_drawdown_log": 0.0, "cdar_5pct_log": 0.0,
            "sharpe_ci_lo": (sharpe_ci[0] if sharpe_ci else None),
            "sharpe_ci_hi": (sharpe_ci[1] if sharpe_ci else None),
            "sortino_ci_lo": None, "sortino_ci_hi": None,
            "max_dd_ci_lo": None, "max_dd_ci_hi": None,
            "block_length": None, "n_bootstrap": None,
        },
        # Calibration: NOT computed in CV mode (engine doesn't capture
        # per-decision (y_true, p_pred)). Use None to signal absence rather
        # than fake zeros.
        "brier": None, "ece": None,
        "reliability": [], "calibration_by_regime": [],
        "roc_auc": None, "pr_auc": None,
        "n_decisions": agg_n_trades, "n_filled": 0,
        "n_actions_approved": 0, "n_actions_rejected": 0,
        "pipeline_state_hash": "",
        # CI / pre-registration block.
        "sharpe_ci": list(sharpe_ci) if sharpe_ci else None,
        "sharpe_seed_band": sharpe_seed_band,
        "brier_ci": None, "ece_ci": None,
        "roc_ci_delong": None, "pr_ci": None,
        "n_trials": 1, "dsr": None,
        "accepted": True, "blocked_reasons": [],
    }


def _build_features(spec: ExperimentSpec):
    """Resolve the features sub-spec to live (base_bar, builder, regime)."""
    from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeCuts, RegimeFeature
    from wagie.features.catalog import default_streaming_features

    base_bar = BaseBarFeatures()
    cat = default_streaming_features()
    if spec.features.n_features is not None and spec.features.n_features > 0:
        cat = cat[: int(spec.features.n_features)]
    elif spec.features.catalog == "minimal":
        cat = cat[:5]
    builder = FeatureBuilder(cat)

    cuts = RegimeCuts(
        feature=spec.features.regime_feature,
        edges=spec.features.regime_edges,
        labels=spec.features.regime_labels,
    )
    return builder, base_bar, RegimeFeature(cuts)


__all__ = ["ExperimentProtocol"]
