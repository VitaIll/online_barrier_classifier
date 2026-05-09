"""Generate the round-flow and backlog-landscape diagrams.

These figures live in `RESEARCH/diagrams/` and are referenced from
`RESEARCH/REPORT_LATEST.md`. Re-run this script after BACKLOG.md changes
significantly (e.g., new tier added).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "RESEARCH" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Diagram 1: round flow with gates
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, *, fc, ec="#222", lw=1.4, fontsize=10, weight="normal"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, zorder=3, color="#111", wrap=True)


def _diamond(ax, x, y, w, h, text, *, fc="#FFE7A8", ec="#7A5C00"):
    pts = [(x + w / 2, y + h), (x + w, y + h / 2), (x + w / 2, y), (x, y + h / 2)]
    diamond = plt.Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=2)
    ax.add_patch(diamond)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=10, weight="bold", color="#3a2a00", zorder=3)


def _arrow(ax, x0, y0, x1, y1, *, color="#444", lw=1.4, label=None, label_offset=(0, 0.12)):
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle="-|>",
                            mutation_scale=14,
                            linewidth=lw,
                            color=color,
                            zorder=1)
    ax.add_patch(arrow)
    if label:
        ax.text((x0 + x1) / 2 + label_offset[0],
                (y0 + y1) / 2 + label_offset[1],
                label, ha="center", va="center", fontsize=9, color=color, style="italic")


def round_flow_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 7.0))

    # palette
    c_init = "#E8F0FE"
    c_research = "#D4E9D9"
    c_impl = "#E5D9F2"
    c_train = "#FCE5D6"
    c_decide = "#FFF1B8"
    c_state = "#E0E0E0"
    c_terminal = {"accept": "#B7E4C7", "iterate": "#FFE0A3", "kill": "#F5B7B1"}

    # Cron trigger (left)
    _box(ax, 0.2, 3.4, 1.2, 0.7, "cron fires\n(daily 06:00)",
         fc="#222", ec="#222", fontsize=9, weight="bold")
    ax.texts[-1].set_color("white")

    # Phase 0
    _box(ax, 1.7, 3.4, 1.6, 0.7, "Phase 0\nInit\nread CONSTITUTION,\nBACKLOG, LEDGER", fc=c_init, fontsize=8.5)

    # Phase 1: parallel research agents
    _box(ax, 3.7, 5.05, 1.6, 0.6, "LITERATURE-\nSCOUT", fc=c_research, fontsize=9)
    _box(ax, 3.7, 4.25, 1.6, 0.6, "CODE-SCOUT", fc=c_research, fontsize=9)
    _box(ax, 3.7, 3.45, 1.6, 0.6, "THEORIST", fc=c_research, fontsize=9)
    _box(ax, 3.7, 2.65, 1.6, 0.6, "(parallel\nspawn)", fc="#F2F2F2", fontsize=8, weight="bold")
    ax.text(4.5, 5.85, "Phase 1 — Research", ha="center", fontsize=11, weight="bold", color="#1B6B33")

    # Phase 2 — implement
    _box(ax, 5.7, 3.4, 1.7, 0.7, "Phase 2\nImplement\nFAST_MODE\n+ tests-first", fc=c_impl, fontsize=8.5)

    # Gate 1: tests
    _diamond(ax, 7.65, 3.25, 1.3, 1.0, "Gate 1\nmake test")

    # Phase 4 — train + eval + MLflow
    _box(ax, 9.25, 3.4, 1.8, 0.7, "Phase 4\nTrain + Eval\nMLflow run\nlog plots+metrics",
         fc=c_train, fontsize=8.5)

    # Gate 2: critic
    _diamond(ax, 11.35, 3.25, 1.3, 1.0, "Gate 2\nCRITIC")

    # Phase 6 decide (3 outcomes branching out)
    _box(ax, 13.0, 5.0, 2.0, 0.6, "accept\ndraft PR → main", fc=c_terminal["accept"], fontsize=9, weight="bold")
    _box(ax, 13.0, 3.4, 2.0, 0.6, "iterate\nback on BACKLOG", fc=c_terminal["iterate"], fontsize=9, weight="bold")
    _box(ax, 13.0, 1.8, 2.0, 0.6, "kill\nKILL_LIST entry", fc=c_terminal["kill"], fontsize=9, weight="bold")

    # Phase 7 — state update (bottom)
    _box(ax, 5.7, 0.7, 5.6, 0.7,
         "Phase 7 — Append LEDGER · update BACKLOG · regenerate REPORT_LATEST · squash-commit on agent/round-NNN-<slug>",
         fc=c_state, fontsize=9)

    # Arrows: cron → init → research (3 fan-out)
    _arrow(ax, 1.4, 3.75, 1.7, 3.75)
    _arrow(ax, 3.3, 3.95, 3.7, 5.35)  # to lit
    _arrow(ax, 3.3, 3.85, 3.7, 4.55)  # to code
    _arrow(ax, 3.3, 3.75, 3.7, 3.75)  # to theorist
    _arrow(ax, 3.3, 3.65, 3.7, 2.95)  # to parallel-spawn note

    # research → implement
    _arrow(ax, 5.3, 5.35, 5.7, 4.0)
    _arrow(ax, 5.3, 4.55, 5.7, 3.85)
    _arrow(ax, 5.3, 3.75, 5.7, 3.75)

    # implement → gate1 → train → gate2
    _arrow(ax, 7.4, 3.75, 7.85, 3.75, label="diff", label_offset=(0, -0.2))
    _arrow(ax, 8.95, 3.75, 9.25, 3.75, label="green", label_offset=(0, -0.2))
    _arrow(ax, 11.05, 3.75, 11.55, 3.75, label="metrics", label_offset=(0, -0.2))

    # gate2 → 3 outcomes
    _arrow(ax, 12.65, 4.05, 13.0, 5.3, color="#1B6B33")
    _arrow(ax, 12.65, 3.75, 13.0, 3.7, color="#7A5C00")
    _arrow(ax, 12.65, 3.45, 13.0, 2.1, color="#7A1B1B")

    # red gate paths (gate1 fail → halt; gate2 veto)
    _arrow(ax, 8.25, 3.25, 8.25, 0.95, color="#7A1B1B", lw=1.2,
           label="red", label_offset=(0.3, 0))
    _arrow(ax, 8.25, 0.95, 5.7, 1.05, color="#7A1B1B", lw=1.2)

    # all outcomes → state update
    _arrow(ax, 13.0, 5.0, 11.3, 1.45, color="#1B6B33", lw=1.0)
    _arrow(ax, 13.0, 3.7, 11.3, 1.4, color="#7A5C00", lw=1.0)
    _arrow(ax, 13.0, 2.1, 11.3, 1.35, color="#7A1B1B", lw=1.0)

    # state update → next round (loop back)
    _arrow(ax, 5.7, 1.05, 0.8, 3.4, color="#444", lw=1.0,
           label="state persists\nbetween rounds", label_offset=(-2.0, 0.3))

    # legend
    legend_handles = [
        mpatches.Patch(facecolor=c_init, edgecolor="#222", label="Init"),
        mpatches.Patch(facecolor=c_research, edgecolor="#222", label="Research (parallel)"),
        mpatches.Patch(facecolor=c_impl, edgecolor="#222", label="Implement"),
        mpatches.Patch(facecolor="#FFE7A8", edgecolor="#7A5C00", label="Gate (blocking)"),
        mpatches.Patch(facecolor=c_train, edgecolor="#222", label="Train + Eval"),
        mpatches.Patch(facecolor=c_terminal["accept"], edgecolor="#222", label="accept"),
        mpatches.Patch(facecolor=c_terminal["iterate"], edgecolor="#222", label="iterate"),
        mpatches.Patch(facecolor=c_terminal["kill"], edgecolor="#222", label="kill"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, -0.02),
              ncol=4, fontsize=9, frameon=False)

    ax.set_xlim(-0.3, 15.5)
    ax.set_ylim(-0.2, 6.4)
    ax.set_axis_off()
    ax.set_title(
        "Autonomous Research-Engineering Loop — One Round\n"
        "Budget: 20 min default, 30 min for backtest rounds. State persists in RESEARCH/ + git.",
        fontsize=12, weight="bold", pad=12,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagram 2: backlog landscape
# ---------------------------------------------------------------------------

def backlog_landscape(path: Path) -> None:
    # Hand-coded from BACKLOG.md (mirror values; if BACKLOG changes, regen).
    rows = [
        # (id, ask_area, priority, expected_alpha, cost, label)
        ("H-001", "Eval (ask 4)",       5, 5.0, 3.0, "Backtest harness"),
        ("H-002", "Eng (ask 5)",        5, 3.0, 1.5, "MLflow logging"),
        ("H-003", "Eng (ask 5)",        5, 2.5, 0.8, "Drop best_params override"),
        ("H-004", "Eng (ask 5)",        5, 3.0, 1.5, "Property tests"),
        ("H-010", "UQ (ask 6)",         5, 4.5, 1.2, "Virtual ensemble UQ"),
        ("H-011", "UQ (ask 6)",         5, 4.0, 1.5, "Conformal prediction"),
        ("H-012", "UQ (ask 6)",         4, 3.0, 0.8, "Isotonic post-hoc cal"),
        ("H-020", "Risk (ask 3)",       4, 4.5, 2.0, "CVaR-tail upweight"),
        ("H-021", "Risk (ask 3)",       4, 4.0, 3.5, "Asym Huber objective"),
        ("H-022", "Risk (ask 3)",       3, 5.0, 2.5, "Meta-labeling layer"),
        ("H-023", "Risk (ask 3)",       3, 5.0, 4.0, "True triple-barrier"),
        ("H-030", "Online (ask 2)",     4, 3.5, 1.0, "SRP classifier"),
        ("H-031", "Online (ask 2)",     4, 3.5, 2.0, "HAT+ALMA stack"),
        ("H-032", "Online (ask 2)",     3, 3.0, 1.0, "Streaming isotonic cal"),
        ("H-033", "Online (ask 2)",     3, 2.5, 0.6, "Online std scaler"),
        ("H-034", "Online (ask 2)",     3, 3.0, 1.0, "ADWIN drift on error"),
        ("H-040", "Features (ask 1)",   4, 3.5, 2.5, "Hurst / DFA / sample-ent"),
        ("H-041", "Features (ask 1)",   3, 3.0, 2.5, "Wavelet energy"),
        ("H-042", "Features (ask 1)",   3, 3.5, 2.5, "HAR-RV residuals"),
        ("H-043", "Features (ask 1)",   3, 2.5, 2.5, "Microprice proxy"),
        ("H-044", "Features (ask 1)",   3, 3.0, 1.0, "Realized higher moments"),
        ("H-045", "Features (ask 1)",   2, 3.5, 4.0, "Multifractal spectrum"),
        ("H-050", "Eng (ask 5)",        4, 2.5, 2.5, "Feature cache"),
        ("H-051", "Eng (ask 5)",        4, 3.0, 2.5, "compute_* registry"),
        ("H-052", "Eng (ask 5)",        3, 3.5, 2.0, "Group ablation"),
        ("H-053", "Eng (ask 5)",        3, 2.0, 1.0, "Profiler pass"),
        ("H-054", "Eng (ask 5)",        3, 1.5, 0.5, "Pre-commit hooks"),
    ]

    palette = {
        "Eval (ask 4)":      "#1B6B33",
        "UQ (ask 6)":        "#0E5174",
        "Risk (ask 3)":      "#7A1B1B",
        "Online (ask 2)":    "#5A2D7A",
        "Features (ask 1)":  "#9C5500",
        "Eng (ask 5)":       "#3A3A3A",
    }

    fig, ax = plt.subplots(figsize=(13, 7.5))

    # Plot bubbles. Size encoded by priority (1..5 → 80..900).
    for hid, area, prio, alpha, cost, label in rows:
        size = 80 + (prio - 1) * 200
        ax.scatter(
            cost, alpha,
            s=size,
            c=palette[area],
            alpha=0.78,
            edgecolors="white",
            linewidths=1.4,
            zorder=3,
        )

    # Annotate items with priority >= 4 to keep label density manageable.
    for hid, area, prio, alpha, cost, label in rows:
        if prio >= 4:
            ax.annotate(
                f"{hid}  {label}",
                xy=(cost, alpha),
                xytext=(8, 4), textcoords="offset points",
                fontsize=8.5, color="#222", zorder=4,
            )

    # Quadrant guides — "do first" zone is high-α / low-cost.
    ax.axhline(3.5, color="#999", linestyle="--", linewidth=0.8, zorder=1)
    ax.axvline(2.0, color="#999", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.4, 4.9, "DO FIRST\nhigh α · low cost", fontsize=10, color="#1B6B33",
            weight="bold", alpha=0.5)
    ax.text(3.2, 4.9, "BIG BETS\nhigh α · high cost", fontsize=10, color="#7A1B1B",
            weight="bold", alpha=0.5)
    ax.text(0.4, 1.6, "QUICK WINS\nlow α · low cost", fontsize=10, color="#3A3A3A",
            weight="bold", alpha=0.5)
    ax.text(3.2, 1.6, "AVOID\nlow α · high cost", fontsize=10, color="#7A5C00",
            weight="bold", alpha=0.5)

    # legends
    area_legend = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=color, markersize=10, label=area)
                   for area, color in palette.items()]
    size_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#444",
               markersize=np.sqrt(80 + (p - 1) * 200) / 2.5,
               label=f"P{p}") for p in (5, 4, 3, 2)
    ]
    leg1 = ax.legend(handles=area_legend, loc="upper right", title="user ask",
                     fontsize=9, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=size_legend, loc="upper right", bbox_to_anchor=(1.0, 0.78),
              title="priority", fontsize=9, title_fontsize=9, frameon=False)

    ax.set_xlabel("Scaffolding cost (rounds; rough estimate)", fontsize=11)
    ax.set_ylabel("Expected α / impact (1–5; rough estimate)", fontsize=11)
    ax.set_xlim(0, 4.6)
    ax.set_ylim(1.0, 5.5)
    ax.set_title(
        "Backlog landscape — 27 hypotheses across 6 user asks\n"
        "Round 1 picks from the top-left (high α, low cost). Bubble size = priority.",
        fontsize=12, weight="bold", pad=10,
    )
    ax.grid(True, axis="both", alpha=0.18, zorder=0)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    round_flow_diagram(OUT / "round_flow.png")
    backlog_landscape(OUT / "backlog_landscape.png")
    print(f"Wrote {OUT/'round_flow.png'}")
    print(f"Wrote {OUT/'backlog_landscape.png'}")


if __name__ == "__main__":
    main()
