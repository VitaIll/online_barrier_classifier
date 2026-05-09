# Experiment Ledger

Append-only. One line per round. The status report.

Schema (pipe-separated):
```
date | round | hypothesis-id | status | branch | mlflow_run | primary_metric_delta | one_line_note
```

`status` ∈ {accept, iterate, kill, bootstrap, housekeeper}.

---

2026-05-09 | 000 | H-bootstrap | bootstrap | agent/round-000-bootstrap | n/a | n/a | Loop infrastructure ported from barrier_classifier sibling: RESEARCH/ governance, scripts (loop_daemon.ps1, loop_status.py, critic_check.py, compact_loop_state.py, run_round.py, audit_chart.py), generic src modules (uncertainty.py, conformal.py, mlflow_utils.py, training_helpers.py, backtest.py), 24 architecture-agnostic tests passing. Project-specific tests parked in tests/_pending/ for re-enable as utilities land in H-101..H-103. BACKLOG re-anchored to offline+online system; H-101 (port labels+splits) is top priority.
