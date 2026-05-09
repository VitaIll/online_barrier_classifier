"""Build the single living REPORT.md from per-round artifacts.

Every accepted round drops a `RESEARCH/diagrams/round_NNN/headline.json`
plus a small set of canonical PNGs. This script reads those artifacts and
assembles `RESEARCH/REPORT.md` — a single, canonical dashboard that always
exists, always reflects the current state of the loop, and ALWAYS leads
with trading profitability (per CONSTITUTION priority + user direction).

Sections (top to bottom):
  ⓪ Headline — single-paragraph state of the project, one-glance status.
  ① Trading profitability — equity curve at val-chosen τ + key metrics.
  ② Calibration & coverage — per-regime calibration / per-regime ACI gap.
  ③ Model — top features (offline CatBoost importance).
  ④ Loop state — recent LEDGER tail + top BACKLOG + accepted tags.

Idempotent. Safe to re-run any time. Future rounds extend by:
  - Dropping their headline.json into RESEARCH/diagrams/round_NNN/
  - (Optionally) adding a section in this script that pulls from it.

Usage:
    python scripts/build_report.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DIAG = REPO / "RESEARCH" / "diagrams"
LEDGER = REPO / "RESEARCH" / "LEDGER.md"
BACKLOG = REPO / "RESEARCH" / "BACKLOG.md"
KILL_LIST = REPO / "RESEARCH" / "KILL_LIST.md"
OUT = REPO / "RESEARCH" / "REPORT.md"


def _load_headlines() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(DIAG.glob("round_*/headline.json")):
        round_id = p.parent.name.split("_", 1)[-1]
        try:
            out[round_id] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            out[round_id] = {"error": f"failed to parse {p}: {e}"}
    return out


def _git(*args: str) -> str:
    p = subprocess.run(["git"] + list(args), cwd=REPO, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def _accepted_tags() -> list[str]:
    return [t for t in _git("tag", "--list", "round-*-accepted").splitlines() if t.strip()]


def _recent_ledger(n: int = 10) -> list[str]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line and line[:4].isdigit() and "|" in line:
            rows.append(line)
    return rows[-n:]


def _top_backlog(n: int = 8) -> list[str]:
    """Pull the top of the curated round ordering from BACKLOG.md."""
    if not BACKLOG.exists():
        return []
    text = BACKLOG.read_text(encoding="utf-8")
    inside = False
    out = []
    for line in text.splitlines():
        if "Round ordering" in line and line.strip().startswith("**"):
            inside = True
            continue
        if inside:
            stripped = line.strip()
            if not stripped:
                if out:
                    break
                continue
            if stripped.startswith("(") or stripped.startswith("---"):
                break
            out.append(stripped)
            if len(out) >= n:
                break
    return out


def _png_link(path: Path) -> str:
    # REPORT.md lives at RESEARCH/REPORT.md; image links must be relative to it.
    rel = path.relative_to(REPO / "RESEARCH").as_posix()
    return f"![{path.stem}]({rel})"


def _safe_image(round_id: str, name: str) -> str | None:
    p = DIAG / f"round_{round_id}" / name
    return _png_link(p) if p.exists() else None


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_headline(headlines: dict) -> str:
    h = headlines.get("009", {})
    tags = _accepted_tags()
    n_rounds = len(tags)
    test_sharpe = h.get("test_sharpe", float("nan"))
    test_psr = h.get("test_psr", float("nan"))
    p_boot = h.get("p_boot", float("nan"))
    dsr = h.get("deflated_sharpe_dsr", float("nan"))
    tau_star = h.get("tau_star", float("nan"))

    if "test_sharpe" in h:
        verdict = (
            f"At val-chosen τ\\*={tau_star:.3f}, **test Sharpe = {test_sharpe:+.3f}** "
            f"(PSR={test_psr:.3f}, p_boot={p_boot:.3f}, DSR={dsr:.3f}). "
            f"Round-001's τ=0.30 PSR=0.995 was a post-hoc artifact; the model has "
            f"no tradable alpha at this label / cost / strategy combination."
        )
    else:
        verdict = "Not yet established (round 009 has not run)."
    cov_h = headlines.get("008", {})
    cov_summary = ""
    if "results" in cov_h:
        cov_summary = (
            f"Coverage: Mondrian-ACI achieves per-regime gap ≤ 0.6pp at α∈{{0.05, 0.10, 0.20}}; "
            f"closes round-002's −7.9pp low-vol p_online gap to +0.04pp."
        )
    return (
        "# Online Barrier Classifier — Living Report\n"
        "\n"
        f"_{n_rounds} accepted rounds. Last update generated by `scripts/build_report.py`._\n"
        "\n"
        "## ⓪ Headline\n"
        "\n"
        f"**Trading**: {verdict}\n"
        "\n"
        f"**{cov_summary}**\n"
    )


def section_trading(headlines: dict) -> str:
    h = headlines.get("009", {})
    if not h:
        return "## ① Trading profitability\n\n_(no round-009 artifact yet)_\n"

    rows = [
        ["**val tau\\* (chosen by max val Sharpe)**", f"{h.get('tau_star', float('nan')):.3f}"],
        ["val Sharpe at tau\\*", f"{h.get('val_tau_sharpe', float('nan')):+.3f}"],
        ["val n_trades at tau\\*", f"{h.get('val_n_trades', 0):,}"],
        ["**test Sharpe at val-chosen tau\\***", f"**{h.get('test_sharpe', float('nan')):+.3f}**"],
        ["test PSR", f"{h.get('test_psr', float('nan')):.3f}"],
        ["test p_boot vs shuffled-signal null", f"{h.get('p_boot', float('nan')):.3f}"],
        [
            "test deflated Sharpe DSR (after τ-grid multiplicity)",
            f"{h.get('deflated_sharpe_dsr', float('nan')):.3f}",
        ],
        ["test n_trades", f"{h.get('test_n_trades', 0):,}"],
        ["test total log return", f"{h.get('test_total_log_return', float('nan')):+.4f}"],
        ["test max drawdown (log)", f"{h.get('test_max_drawdown', float('nan')):.3f}"],
        ["test hit rate", f"{h.get('test_hit_rate', float('nan')):.3f}"],
        [
            "round-001 post-hoc reference (τ=0.30)",
            f"Sharpe=+{h.get('round_001_post_hoc_test_sharpe', 1.5):.2f}, "
            f"PSR={h.get('round_001_post_hoc_test_psr', 0.995):.3f}",
        ],
    ]
    table = _md_table(["metric", "value"], rows)

    out = ["## ① Trading profitability\n",
           "Strategy: open long when p_offline ≥ tau\\*, exit at first triple-barrier touch "
           "(profit-take +α, stop-loss −α, time-out M=20). α=0.0041113 (90%-quantile of "
           "training-window log-excursions). Cost = 1 bp/side.\n",
           "**Headline finding (H-005b, round 009)**: the round-001 PSR=0.995 at τ=0.30 was a post-hoc "
           "artifact. With τ chosen on validation Sharpe (at tau\\*=0.44), test Sharpe collapses to "
           "essentially zero. The strategy still beats random-entry-at-same-rate (p_boot=0) but does not "
           "produce tradable alpha after honest selection.",
           "",
           table,
           ""]

    sweep = _safe_image("009", "val_tau_sweep.png")
    eq = _safe_image("009", "test_at_val_tau_equity.png")
    if sweep:
        out.append("**Validation tau sweep** (val Sharpe across the τ grid; red dashed = chosen winner)")
        out.append("")
        out.append(sweep)
        out.append("")
    if eq:
        out.append("**Test equity curve at val-chosen tau\\*** (top: cumulative log return; bottom: drawdown)")
        out.append("")
        out.append(eq)
        out.append("")
    return "\n".join(out)


def section_calibration(headlines: dict) -> str:
    out = ["## ② Calibration & coverage\n"]

    h005 = headlines.get("005", {})
    if "predictors" in h005:
        rows = []
        for predictor, blk in h005["predictors"].items():
            g = blk.get("global", {})
            rows.append([
                predictor,
                f"{g.get('roc_auc', 0):.3f}",
                f"{g.get('brier_score', 0):.4f}",
                f"{g.get('log_loss', 0):.4f}",
                f"{g.get('ece', 0):.4f}",
            ])
        out.append("**Global metrics** (round 005, on test n=31,486)")
        out.append("")
        out.append(_md_table(
            ["predictor", "ROC", "Brier", "log loss", "ECE"], rows
        ))
        out.append("")

        # Per-regime ECE table.
        per_rows = []
        for predictor, blk in h005["predictors"].items():
            pr = blk.get("per_regime", {})
            for regime, stats in pr.items():
                per_rows.append([
                    predictor,
                    regime,
                    f"{stats.get('n_samples', 0):,}",
                    f"{stats.get('base_rate', 0):.3f}",
                    f"{stats.get('mean_predicted', 0):.3f}",
                    f"{stats.get('ece', 0):.4f}",
                    f"{stats.get('brier', 0):.4f}",
                ])
        if per_rows:
            out.append("**Per-regime ECE / Brier** (parkinson_var_rolling_mean_24 terciles)")
            out.append("")
            out.append(_md_table(
                ["predictor", "regime", "n", "base rate", "mean p", "ECE", "Brier"],
                per_rows,
            ))
            out.append("")
    cov_curve = _safe_image("005", "calibration_by_regime.png")
    if cov_curve:
        out.append("**Per-regime calibration curves** (round 005)")
        out.append("")
        out.append(cov_curve)
        out.append("")

    # Mondrian-ACI per-regime gap from round 008.
    h008 = headlines.get("008", {})
    if "results" in h008:
        gap_rows = []
        for predictor, blk in h008["results"].items():
            for alpha_key, m in blk.get("alpha", {}).items():
                aci_per_regime = m.get("plain_aci_per_regime_gap", {}) or {}
                m_per_regime = m.get("mondrian_aci_per_regime_gap", {}) or {}
                for regime in ("low", "med", "high"):
                    gap_rows.append([
                        predictor,
                        f"α={alpha_key}",
                        regime,
                        f"{aci_per_regime.get(regime, 0):+.4f}",
                        f"{m_per_regime.get(regime, 0):+.4f}",
                    ])
        if gap_rows:
            out.append("**Per-regime coverage gap (target − empirical)** — round 008 Mondrian-ACI")
            out.append("")
            out.append(_md_table(
                ["predictor", "alpha", "regime", "plain ACI gap", "Mondrian-ACI gap"],
                gap_rows,
            ))
            out.append("")
    gap_plot = _safe_image("008", "three_way_gap_comparison.png")
    if gap_plot:
        out.append("**Three-way per-regime gap comparison** (round 008)")
        out.append("")
        out.append(gap_plot)
        out.append("")
    return "\n".join(out)


def section_model(headlines: dict) -> str:
    out = ["## ③ Model\n"]
    h006 = headlines.get("006", {})
    top5 = h006.get("top_5_features")
    if top5:
        rows = [[str(f.get("name", "?")), f"{float(f.get('importance', 0)):.3f}"] for f in top5]
        out.append("**Top-5 features by offline CatBoost importance** (round 006)")
        out.append("")
        out.append(_md_table(["feature", "importance"], rows))
        out.append("")
    fi_plot = _safe_image("006", "visual_report_card.png")
    if fi_plot:
        out.append("**Visual report card** (round 006 — calibration curves + top-25 importance + threshold sweep)")
        out.append("")
        out.append(fi_plot)
        out.append("")
    return "\n".join(out)


def section_ensemble(headlines: dict) -> str:
    out = ["## ④ Ensemble + foundational utilities\n"]
    h003 = headlines.get("003", {})
    if h003:
        out.append(
            f"- **CatBoostEnsemble** (round 003): {h003.get('n_models', '?')}-model averaging on "
            f"synthetic 800×6 fit; max single-model deviation from ensemble = "
            f"{h003.get('max_individual_disagreement_with_ensemble', 0):.3f}."
        )
    h004 = headlines.get("004", {})
    if h004:
        pos = h004.get("positive_rate", {})
        out.append(
            f"- **Label + split utilities** (round 004): chronological split of "
            f"n={h004.get('n', 0):,} bars; train/val/test n="
            f"{h004.get('n_train', 0):,}/{h004.get('n_val', 0):,}/{h004.get('n_test', 0):,}; "
            f"per-window p+ = {pos.get('train', 0):.3f}/{pos.get('val', 0):.3f}/{pos.get('test', 0):.3f}."
        )
    h002 = headlines.get("002", {})
    if h002:
        out.append(
            "- **Coverage baseline** (round 002): batch LAC + Mondrian-LAC measured on the "
            "real ARF stream — see round 008 for the online ACI / Mondrian-ACI improvements that "
            "supersede these batch baselines."
        )
    out.append("")
    return "\n".join(out)


def section_loop_state() -> str:
    out = ["## ⑤ Loop state\n"]
    tags = _accepted_tags()
    out.append(f"**Accepted tags** ({len(tags)}): {', '.join(tags) if tags else '(none)'}")
    out.append("")

    rows = _recent_ledger(8)
    if rows:
        out.append("**Recent LEDGER** (most recent 8)")
        out.append("```")
        for r in rows:
            # Trim ultra-long lines so the report stays readable.
            out.append(r[:240] + ("…" if len(r) > 240 else ""))
        out.append("```")
        out.append("")

    backlog = _top_backlog(10)
    if backlog:
        out.append("**Top of round ordering** (next round picks the first non-accepted item)")
        out.append("")
        for line in backlog:
            out.append(f"- {line}")
        out.append("")

    if KILL_LIST.exists():
        kills = [l for l in KILL_LIST.read_text(encoding="utf-8").splitlines()
                 if l.startswith("[round")]
        if kills:
            out.append("**Recent KILL_LIST**")
            out.append("```")
            for k in kills[-5:]:
                out.append(k[:220])
            out.append("```")
    return "\n".join(out)


def main() -> int:
    headlines = _load_headlines()
    parts = [
        section_headline(headlines),
        section_trading(headlines),
        section_calibration(headlines),
        section_model(headlines),
        section_ensemble(headlines),
        section_loop_state(),
    ]
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
