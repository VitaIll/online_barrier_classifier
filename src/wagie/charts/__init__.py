"""wagie.charts — the SINGLE chart battery.

``ChartBattery.render_all(result, out_dir)`` produces every chart this repo
supports, in a consistent visual theme. Calibration leads, then trading.
No bespoke per-experiment charts: extend the battery instead.

Public API:
    from wagie.charts import ChartBattery
    paths = ChartBattery().render_all(engine_result, out_dir)
"""

from __future__ import annotations

# Force a non-interactive backend before any submodule imports pyplot.
import matplotlib as _matplotlib
_matplotlib.use("Agg", force=True)

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from wagie.engine import EngineResult

from .calibration import probability_histogram, reliability_diagram
from .theme import PALETTE, apply_theme, figsize
from .trading import drawdown_chart, equity_curve, pnl_distribution


@dataclass
class ChartBattery:
    """The ONE chart battery. Configure once, render against any EngineResult.

    Backend: matplotlib (loaded lazily — chart code only imports it when called).
    """

    n_calibration_bins: int = 10

    def render_all(
        self,
        result: EngineResult,
        out_dir: Path,
        *,
        y_true: Optional[Sequence[int]] = None,
        p_pred: Optional[Sequence[float]] = None,
    ) -> dict[str, Path]:
        """Render every default chart. Returns name → path map."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out: dict[str, Path] = {}

        pnl = [float(f.pnl_log_net) for f in result.ledger.fills]
        out["equity_curve"] = equity_curve(pnl, out_dir / "01_equity_curve.png")
        out["drawdown"] = drawdown_chart(pnl, out_dir / "02_drawdown.png")
        out["pnl_distribution"] = pnl_distribution(pnl, out_dir / "03_pnl_distribution.png")

        # Calibration data: prefer caller-supplied; otherwise pull from the
        # engine's streaming capture so charts reflect what metrics see.
        if y_true is None and hasattr(result, "label_history"):
            y_true = result.label_history
        if p_pred is None and hasattr(result, "p_online_history"):
            p_pred = result.p_online_history

        if y_true is not None and p_pred is not None and len(y_true) > 0:
            out["reliability"] = reliability_diagram(
                y_true, p_pred, out_dir / "04_reliability.png",
                n_bins=self.n_calibration_bins,
            )
            out["p_histogram"] = probability_histogram(
                p_pred, out_dir / "05_p_histogram.png",
            )

        return out


__all__ = [
    "ChartBattery",
    "PALETTE", "apply_theme", "figsize",
    "reliability_diagram", "probability_histogram",
    "equity_curve", "drawdown_chart", "pnl_distribution",
]
