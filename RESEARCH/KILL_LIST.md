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

[round 018] H-209 (single-seed virtual-ensemble Cantelli sizing): ITERATE. Cantelli sizing with σ_epistemic from single SGLB CatBoost (n_virtual=10) on round-017's c_stop=∞ + tau=0.20 baseline produced k\*=0 winning val (val Sharpe -5.33, all k>0 collapsed to ≤54 trades or 0). Test Sharpe at k\*=0 = -5.55, ΔvsR017 binary baseline = -6.17. Three diagnostic findings: (i) `|p_offline_cached - p_mean_vens| max = 0.40` — virtual-ensemble mean differs hugely from full predict_proba because n=10 sub-ensembles each have only ~62 trees vs full 625; (ii) σ_epistemic mean = 0.47 — much higher than p_offline mean 0.21 — Cantelli p_lower = max(0, p − k·σ) collapses to 0 for most bars at k > 0.5; (iii) per-σ-decile chart shows mean PnL is *inverse-monotone* in σ_epistemic (low-σ deciles -0.0005, high-σ deciles -0.00002) — opposite of expected, suggesting single-seed σ from boosting trajectory is anti-informative for trade selection on this data; (iv) cost-on-full-position model (per backtest mandate §A.3) penalises sized-down strategies — small sizes pay full cost. **The proper H-209 test requires a 3-seed CatBoostEnsemble** (per BACKLOG H-209 spec): each seed trained with langevin=True, virtual-ensemble decomposition pooled via the ensemble-of-ensembles formula in `predict_with_decomposed_uq`. Queued as H-209-v2 (depends on 3-seed retrain — separate round). Round-018 artefacts (`RESEARCH/diagrams/phase_A/round_018/`) preserved as the kill diagnosis. **Round 019 H-210 (virtual-ensemble agreement gate) SKIPPED for the same reason** — identical single-seed inadequacy; queued as H-210-v2 (depends on 3-seed ensemble).
