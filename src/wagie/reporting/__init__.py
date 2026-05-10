"""wagie.reporting — the SINGLE report format.

`Report.render(spec, metrics, charts, out_path)` writes a Markdown report
with calibration leading, then trading, then coverage. No bespoke per-
experiment templates: extend this renderer.

Public API:
    from wagie.reporting import Report
    Report().render(spec, metrics_report, chart_paths, out_path)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional


@dataclass
class Report:
    """The ONE report renderer. Markdown only — figures are embedded as
    relative-path images so the report is portable inside the run dir."""

    title: str = "wagie experiment report"
    show_provenance: bool = True

    def render(
        self,
        spec_dict: Mapping,
        metrics: Mapping,
        chart_paths: Mapping[str, Path],
        out_path: Path,
        *,
        run_id: Optional[str] = None,
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        run_id = run_id or "—"

        lines: list[str] = []
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**run_id:** `{run_id}`  ")
        lines.append(f"**generated:** {ts}  ")
        if self.show_provenance:
            sh = metrics.get("pipeline_state_hash", "")
            if sh:
                lines.append(f"**state_hash:** `{sh}`  ")
        lines.append("")

        # === 1. Calibration (lead) ===
        lines.append("## 1. Calibration")
        lines.append("")
        brier_val = metrics.get('brier')
        ece_val = metrics.get('ece')
        if isinstance(brier_val, (int, float)):
            lines.append(f"- **Brier**: `{brier_val:.5f}`")
        else:
            lines.append("- **Brier**: `n/a` (mode does not capture per-decision predictions)")
        if isinstance(ece_val, (int, float)):
            lines.append(f"- **ECE**: `{ece_val:.5f}`")
        else:
            lines.append("- **ECE**: `n/a` (mode does not capture per-decision predictions)")
        if "reliability" in chart_paths:
            lines.append("")
            lines.append(self._image("Reliability diagram", chart_paths["reliability"], out_path))
        if "p_histogram" in chart_paths:
            lines.append("")
            lines.append(self._image("Predicted-probability distribution",
                                      chart_paths["p_histogram"], out_path))
        lines.append("")

        # === 2. Trading ===
        lines.append("## 2. Trading")
        lines.append("")
        t = metrics.get("trading", {}) or {}
        lines.append(f"- **n_trades**: `{t.get('n_trades', 0)}`  "
                     f"(TP={t.get('n_tp', 0)} / SL={t.get('n_sl', 0)} / "
                     f"timeout={t.get('n_timeout', 0)})")
        lines.append(f"- **Sharpe (annualized)**: `{t.get('sharpe', 0.0):+.3f}`")
        lines.append(f"- **PSR**: `{t.get('probabilistic_sharpe', 0.0):.3f}`")
        lines.append(f"- **Sortino**: `{t.get('sortino', 0.0):+.3f}`")
        lines.append(f"- **hit rate**: `{t.get('hit_rate', 0.0):.3f}`")
        lines.append(f"- **profit factor**: `{t.get('profit_factor', 0.0):.3f}`")
        lines.append(f"- **max drawdown (log)**: `{t.get('max_drawdown_log', 0.0):.4f}`")
        lines.append(f"- **CDaR 5% (log)**: `{t.get('cdar_5pct_log', 0.0):.4f}`")
        lines.append(f"- **total log return**: `{t.get('total_log_return', 0.0):+.4f}` "
                     f"({t.get('total_pct_return', 0.0):+.2%})")
        if "equity_curve" in chart_paths:
            lines.append("")
            lines.append(self._image("Equity curve", chart_paths["equity_curve"], out_path))
        if "drawdown" in chart_paths:
            lines.append("")
            lines.append(self._image("Drawdown", chart_paths["drawdown"], out_path))
        if "pnl_distribution" in chart_paths:
            lines.append("")
            lines.append(self._image("PnL distribution", chart_paths["pnl_distribution"], out_path))
        lines.append("")

        # === 3. Coverage ===
        lines.append("## 3. Conformal coverage")
        lines.append("")
        cov = metrics.get("coverage", []) or []
        if cov:
            lines.append("| α | empirical | target | gap | n |")
            lines.append("|---|---|---|---|---|")
            for c in cov:
                lines.append(
                    f"| {c['alpha']:.2f} | {c['empirical']:.3f} | "
                    f"{c['target']:.3f} | {c['gap']:+.3f} | {c['n']} |"
                )
        else:
            lines.append("_no coverage data captured_")
        if "coverage_bars" in chart_paths:
            lines.append("")
            lines.append(self._image("Coverage", chart_paths["coverage_bars"], out_path))
        if "quantile_drift" in chart_paths:
            lines.append("")
            lines.append(self._image("Quantile drift", chart_paths["quantile_drift"], out_path))
        lines.append("")

        # === 4. Operational ===
        lines.append("## 4. Operational")
        lines.append("")
        lines.append(f"- **n_decisions**: `{metrics.get('n_decisions', 0)}`")
        lines.append(f"- **n_filled**: `{metrics.get('n_filled', 0)}`")
        lines.append(f"- **actions approved / rejected**: "
                     f"`{metrics.get('n_actions_approved', 0)}` / "
                     f"`{metrics.get('n_actions_rejected', 0)}`")
        roc_val = metrics.get('roc_auc')
        pr_val = metrics.get('pr_auc')
        if isinstance(roc_val, (int, float)):
            lines.append(f"- **ROC-AUC** (diagnostic): `{roc_val:.3f}`")
        else:
            lines.append("- **ROC-AUC** (diagnostic): `n/a`")
        if isinstance(pr_val, (int, float)):
            lines.append(f"- **PR-AUC** (diagnostic): `{pr_val:.3f}`")
        else:
            lines.append("- **PR-AUC** (diagnostic): `n/a`")
        lines.append("")

        # === 5. Spec ===
        lines.append("## 5. Spec")
        lines.append("")
        lines.append("```yaml")
        lines.append(self._yaml_dump(spec_dict))
        lines.append("```")
        lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    def _image(self, alt: str, src: Path, anchor: Path) -> str:
        try:
            rel = Path(src).resolve().relative_to(Path(anchor).parent.resolve())
            return f"![{alt}]({rel.as_posix()})"
        except ValueError:
            return f"![{alt}]({Path(src).as_posix()})"

    def _yaml_dump(self, d: Mapping) -> str:
        try:
            import yaml
            return yaml.safe_dump(dict(d), sort_keys=False, default_flow_style=False).rstrip()
        except Exception:
            import json
            return json.dumps(dict(d), indent=2, default=str)


__all__ = ["Report"]
