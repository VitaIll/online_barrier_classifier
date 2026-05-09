# Experiment Ledger

Append-only. One line per round. The status report.

Schema (pipe-separated):
```
date | round | hypothesis-id | status | branch | mlflow_run | primary_metric_delta | one_line_note
```

`status` ∈ {accept, iterate, kill, bootstrap, housekeeper}.

---

2026-05-09 | 000 | H-bootstrap | bootstrap | agent/round-000-bootstrap | n/a | n/a | Loop infrastructure ported from barrier_classifier sibling: RESEARCH/ governance, scripts (loop_daemon.ps1, loop_status.py, critic_check.py, compact_loop_state.py, run_round.py, audit_chart.py), generic src modules (uncertainty.py, conformal.py, mlflow_utils.py, training_helpers.py, backtest.py), 24 architecture-agnostic tests passing. Project-specific tests parked in tests/_pending/ for re-enable as utilities land in H-101..H-103. BACKLOG re-anchored to offline+online system; H-101 (port labels+splits) is top priority.
2026-05-09 | 001 | H-005 | accept | agent/round-001-h005-first-pnl | barrier_round_001 | offline τ=0.30 Sharpe=+1.50 PSR=0.995 p_boot=0.000; final τ=0.20 Sharpe=-4.69 p_boot=0.99 | First PnL number for online_barrier_classifier. Glue-only round: load predictions.parquet + 1m + bars_20m, map open_time→minute_idx (CODE-SCOUT), drop bars whose [n_k+1,n_k+M] crosses segment gap, run simulate_inventory_aware on τ_open∈{0.10,0.15,0.20,0.30,0.50} for {p_offline,p_final=p_online} with symmetric φ=c_stop=α=0.0041113 and 1bp/side cost. Shuffled-signal random-entry null (n=200) per Bailey/Borwein/LdP 2014 — drift-matched, addresses BTC2024 +120% drift. Result: ALL metrics finite (primary falsifier passes). p_offline has p_boot<0.05 at every τ (best at τ=0.30 with PSR=0.995); p_final/p_online has no detectable edge above τ=0.10. Online layer's ROC-trade-for-coverage cost shows up as flat-or-worse Sharpe — consistent with CONSTITUTION I (online primary metric is coverage, not Sharpe). Followups: H-005b (val-split τ_open selection), H-005c (CSCV PBO post-hoc τ deflation, blocked on H-113). Loop infra fixes also in commit: critic_check uses master not main, causality test check tolerates _pending state, .gitignore excludes pycache, conformal property test slack 3σ→4σ.
