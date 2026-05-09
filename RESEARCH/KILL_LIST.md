# Kill List

Append-only record of falsified or rejected hypotheses. Prevents re-litigation.

Schema:
```
[round NNN] H-NNN: <one-line claim> — <one-line reason>
```

---

(empty — RETRACTED: round 002 entry removed because the round actually completed legitimate work (H-201 accept, see commit 2ef7bfc). The "kill" was a false alarm during discipline rollout — the branch was killed but the work was recovered from the session JSONL and committed to master.)

[round 010] H-024 in_progress (no-stop variant, c_stop=10·α): never accepted; superseded by round-015's unified harness which supports `c_stop=∞` natively. Script `scripts/round_010_no_stop_backtest.py` and partial `RESEARCH/diagrams/round_010/` deleted under the round-015 cleanup.

[round 011] phase-A-r1 (committed 1e82da4): SUPERSEDED-by-015. Architectural error — `combined_avg_tau` averaged `p_offline` and `p_online` as if they were sibling signals. The two-layer architecture says `p_online` is the OUTPUT of the online layer that has already consumed `p_offline` as input. Strategy + diagrams + unified prediction parquets deleted; the offline + online + null rows of round-011 are now part of the round-015 corrected table. Reproduces the round-009 sanity floor (test Sharpe=−0.089 at τ=0.44) but with the wrong combined strategy alongside.

[round 012] phase-A-r2 (committed 125a44c): SUPERSEDED-by-015. Two architectural errors: (i) `combined_stacked_tau` regressed `y` on `(p_offline, p_online)` — i.e., regressed the label on parent + child, which is invalid; (ii) `src/inference.py::predict()` drove the Mondrian-ACI conformal stream off `p_offline` instead of `p_online`. Fixed in round-015. Strategy code (`fit_stacker`, `combined_stacked_tau`) deleted; diagrams + unified parquets deleted. Walk-forward and CSCV machinery in `src/backtest.py` is preserved (architecture-agnostic; round-015 reuses).

[round 013] phase-A-r3 (never committed): KILLED. The runner `scripts/phase_A_round_013.py` was started but interrupted by the user's research-planning redirect; no LEDGER entry, no commit. Script + partial `RESEARCH/diagrams/phase_A/round_013/` deleted under round-015 cleanup. Round-015 covers the same scope (CSCV + Phase A report) under the corrected architecture.
