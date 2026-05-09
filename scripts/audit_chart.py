"""Audit chart: user-ask coverage across the 6 rounds executed so far."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "RESEARCH" / "diagrams" / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    asks = [
        ("1. Features (literature, time-invariant)", 0,
         "0/5 hypotheses started (H-040..H-045 all queued)"),
        ("2. Online model improvements (River)", 0,
         "0/5 hypotheses started (H-030..H-034 all queued)"),
        ("3. Risk-averse weighting (beyond H-007)", 5,
         "Existing scheme retained; no new tail-up-weighting (H-020..H-023 queued)"),
        ("4. Sophisticated evaluation / backtest", 45,
         "Harness + UQ + conformal landed but no real-data run (H-005 queued)"),
        ("5. Engineering / production / tests", 65,
         "103 tests, MLflow scaffolded but unused, no CI, no pre-commit hook"),
        ("6. CatBoost virtual ensemble UQ", 75,
         "Wired + validated; not in notebook (H-014 blocked on H-007)"),
        ("7. MLflow experiment tracking", 25,
         "Helpers exist, never invoked from real training run"),
    ]
    n = len(asks)
    coverage = np.array([a[1] for a in asks])
    labels = [a[0] for a in asks]
    notes = [a[2] for a in asks]
    colors = [
        "#7A1B1B" if c < 20 else
        "#7A5C00" if c < 50 else
        "#1B6B33" if c >= 70 else
        "#0E5174"
        for c in coverage
    ]

    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    bars = ax.barh(np.arange(n), coverage, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=1.2)
    for i, (bar, c, note) in enumerate(zip(bars, coverage, notes)):
        ax.text(min(c + 1.5, 92), i, f"{c}%", va="center",
                fontsize=10, weight="bold", color="#222")
        ax.text(101, i, note, va="center", fontsize=8.5, color="#444")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 220)
    ax.set_xlabel("Approximate coverage (% of work done so far against this ask)", fontsize=10)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.axvline(50, color="#999", linestyle="--", linewidth=0.6)
    ax.set_title(
        "User-ask coverage after 6 rounds — three asks at 0% start\n"
        "(features + online learning + risk-aware weighting completely untouched)",
        fontsize=12, weight="bold", pad=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "ask_coverage.png", dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Tier-of-work chart: what the rounds actually did
    rounds = ["R-000\nbootstrap", "R-001\nbacktest\nharness", "R-002\nliterature\nindex",
              "R-003\nMLflow+\nhypothesis", "R-004\nUQ", "R-005\nretrain", "R-006\nconformal"]
    tiers = [
        ("Infrastructure", [1, 0, 1, 1, 0, 0, 0]),
        ("Modeling-related", [0, 1, 0, 0, 1, 1, 1]),
        ("Real-data eval", [0, 0, 0, 0, 0, 0, 0]),
        ("Online learning", [0, 0, 0, 0, 0, 0, 0]),
        ("New features", [0, 0, 0, 0, 0, 0, 0]),
    ]
    palette = ["#7A5C00", "#0E5174", "#1B6B33", "#5A2D7A", "#7A1B1B"]

    fig, ax = plt.subplots(figsize=(13.5, 4.5))
    bottom = np.zeros(len(rounds))
    for (name, vals), color in zip(tiers, palette):
        v = np.array(vals)
        ax.bar(rounds, v, bottom=bottom, label=name, color=color, alpha=0.85,
               edgecolor="white", linewidth=1.2)
        bottom = bottom + v
    ax.set_ylabel("Type of work in round (1.0 = sole focus)", fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9, ncol=5, frameon=False, bbox_to_anchor=(1.0, 1.18))
    ax.set_title(
        "Where the 6 rounds actually landed — heavy infrastructure, no real-data eval, "
        "no online learning, no new features",
        fontsize=12, weight="bold", pad=22,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUT / "round_tier_distribution.png", dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Round-007 hardening: before/after gap closure chart
    issues_before = [
        ("Self-trigger fragility (no daemon)", 0),
        ("No status/observability command", 0),
        ("No code-level CRITIC checklist", 0),
        ("No artifact compaction", 0),
        ("MLflow scaffolded but unused", 25),
        ("BACKLOG mis-prioritized vs user goals", 0),
        ("ROUND_TEMPLATE not resilient (re-arm last)", 0),
        ("No round wall-clock hard cap", 0),
    ]
    issues_after = [
        ("Self-trigger fragility (no daemon)", 95),         # loop_daemon.ps1
        ("No status/observability command", 90),            # loop_status.py
        ("No code-level CRITIC checklist", 95),             # critic_check.py
        ("No artifact compaction", 95),                     # compact_loop_state.py
        ("MLflow scaffolded but unused", 25),               # still pending — H-014
        ("BACKLOG mis-prioritized vs user goals", 80),      # H-005 first; Tier 0 added
        ("ROUND_TEMPLATE not resilient (re-arm last)", 95), # Phase A first
        ("No round wall-clock hard cap", 90),               # daemon timeout
    ]
    n2 = len(issues_before)
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    y = np.arange(n2)
    bar_w = 0.38
    before = np.array([v[1] for v in issues_before])
    after = np.array([v[1] for v in issues_after])
    labels = [v[0] for v in issues_before]
    ax.barh(y - bar_w / 2, before, bar_w, color="#7A1B1B", alpha=0.8,
            edgecolor="white", linewidth=1.0, label="before round-007")
    ax.barh(y + bar_w / 2, after, bar_w, color="#1B6B33", alpha=0.85,
            edgecolor="white", linewidth=1.0, label="after round-007")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(b + 1.5, i - bar_w / 2, f"{b}%", va="center", fontsize=9, color="#7A1B1B")
        ax.text(a + 1.5, i + bar_w / 2, f"{a}%", va="center", fontsize=9, color="#1B6B33", weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Approximate gap-closure (% of issue resolved)", fontsize=10)
    ax.set_title(
        "Round-007 audit remediation — 7 of 8 audit items closed; "
        "1 item (MLflow wiring) deferred to H-014",
        fontsize=12, weight="bold", pad=8,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "round_007_remediation.png", dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote audit charts to {OUT}")


if __name__ == "__main__":
    main()
