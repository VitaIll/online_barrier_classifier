# Online Barrier Classifier — Living Report

_Single source of truth. Hand-maintained as of the post-conformal-removal wave (2026-05-10).
The previous `scripts/build_report.py` regenerator referenced deleted scripts and is no
longer canonical. Update this file in place at the end of any round that changes
architecture or headline numbers._

---

## (-1) Headline (post-conformal-removal — current state)

**The Mondrian-ACI conformal layer has been removed.** Sibling ARCH owns the cleanup. The
new strategy is `ThresholdGate(p_online >= tau)`. Calibration of `p_online` (Brier, ECE
with bootstrap CI) is the headline metric; trading metrics (Sharpe, drawdown) are reported
alongside but are downstream of the calibration claim.

**Round-031's `+20.86 ann-Sharpe` low-vol-gate finding is REVOKED pending wagie replay.**
The legacy `simulate_inventory_aware_sized` harness from round-031 contained a 19-minute
look-ahead between feature computation and trade entry (see `RESEARCH/HEALTH_OF_RESULTS.md`
for the full revocation list and reasoning). The replay spec lives at
`experiments/replay_r031_low_vol_gate.yaml`; until that spec runs and the bootstrap-CI
lower bound on the headline metric beats `predicted_effect_min: 0.5` ann-Sharpe with
`min_n_trades: 200`, the +20.86 number must NOT appear as a quoted result anywhere.

**Replays under the wagie harness completed**: zero. The `BEFORE_vs_AFTER.csv` artefact
named in the round-017 replay roadmap does not yet exist. See HEALTH_OF_RESULTS.md
"What's been replayed" section.

## 0. Architecture (binding — see [`docs/architecture.md`](../docs/architecture.md))

```
            +- Layer 1 (offline) -+    +- Layer 2 (online — system output) -+
20m bars -> | CatBoost (frozen)   | -> | River ARFClassifier consumes        | -> p_online
            |  -> p_offline       |    | (selected_features, p_offline?)     |    (calibrated)
            +---------------------+    | -> p_online                         |
                                       +-------------------------------------+
                                                    |
                                                    v
                                       ThresholdGate(p_online >= tau)
                                       (NO separate conformal layer; calibration
                                        comes from the ARF bagging + ADWIN drift)
```

**`p_offline` is INPUT to the online layer; `p_online` is the system OUTPUT.** Anything
that averages or stacks `(p_offline, p_online)` is architecturally invalid (it regresses
the child onto its parent) and is forbidden by the strategy registry. This invariant
survives the conformal-removal wave intact.

**What changed in this wave**: the previous `MondrianACICalibrator` Pipeline stage and the
`PureConformalGate(in_set_alpha == 1)` strategy are gone. The default strategy is now
`ThresholdGate(p_online >= tau)` with `tau` chosen by validation Sharpe / Brier sweep.

## 1. Headline metric (calibration leads)

**Calibration of `p_online` is the system's primary deliverable.** The accept-gate is:

| Metric | Schema | Bootstrap CI scheme |
|---|---|---|
| Brier(`p_online`) | per-bar; per-regime | stationary block bootstrap (Politis-White block length) |
| ECE(`p_online`)   | 10 equal-width bins | stratified bootstrap within bins; debiased per Roelofs et al. 2022 |
| Coverage gap (per-regime) | `1 - alpha - empirical_coverage` per tercile, per alpha | per-regime stratified bootstrap |
| Reliability diagram | `charts/reliability_diagram.png` per run | per-bin Wilson interval |
| Sharpe (downstream) | per-trade with stationary block bootstrap CI | block length via Politis-White |

Sibling RIGOR is shipping the bootstrap-CI library and the accept-gate enforcement on
`(hypothesis_id, predicted_effect_min, min_n_trades)`. Until that lands, headline numbers
continue to be reported as point-estimates with a "CI pending" annotation.

## 2. Headline numbers — DEFERRED

No round under the wagie harness has yet produced calibration headline numbers with proper
bootstrap CIs. The first one will come from running `experiments/baseline.yaml` end-to-end
on the canonical data slice (BTCUSDT 2023-01..2025-12) once sibling ARCH has finalized the
spec and sibling RIGOR has shipped the CI library.

**Do not quote pre-wagie numbers as accepted.** The legacy `phase_A_round_015.py` table
(test Sharpe = -0.089 for `baseline_offline_tau`) was the last "accepted" claim; that
script no longer exists in the repo (deleted in commit `61ce420`).

## 3. Loop state

**Last accepted round (under the legacy harness)**: round-031 (low-vol-gate) on 2026-05-09.
**Status of all rounds 015..031 under the new wagie harness**: REVOKED pending replay.

See `RESEARCH/HEALTH_OF_RESULTS.md` for the full revocation list. The revocation does not
mean the model has no edge — it means the legacy backtest's 19-minute look-ahead has not
yet been ruled out as the source of the apparent edge.

**Next priority replay** (per HEALTH_OF_RESULTS.md): R-031 low-vol gate, expressed in the
new spec format at `experiments/replay_r031_low_vol_gate.yaml`. Expectation: the +20.86
ann-Sharpe collapses to near zero or negative once honest execution timing is enforced.
Falsification gate: bootstrap-CI lower bound on ann-Sharpe must exceed `+0.5` with at
least 200 trades for the round to accept.

## 4. Reproduction

```bash
git clone <repo>
cd online_barrier_classifier
pip install -e ".[dev]"
# Generate the synthetic stand-in (or point baseline.yaml at real data):
# (see README.md "Quick start" for the one-liner)
wagie experiment run experiments/baseline.yaml
```

Output lives in `artifacts/runs/<run_id>/`:
- `metrics.json` — calibration leads (`brier`, `ece`, per-regime), then `trading` block.
- `charts/reliability_diagram.png` — produced before any other chart.
- `report.md` — per-run dashboard.

## 5. Spec catalog (committed YAMLs)

| Spec | Purpose |
|---|---|
| `experiments/baseline.yaml` | Default — ARF only, no offline; sibling ARCH owns. |
| `experiments/baseline_with_offline.yaml` | Two-layer: CatBoost -> ARF -> ThresholdGate. Demo for when a `.cbm` exists. |
| `experiments/sweep_alpha.yaml` | Five-spec sweep over `wagie.strategy.tau` (the new gating knob). Replaces the legacy `aci.alphas` sweep. |
| `experiments/replay_r031_low_vol_gate.yaml` | Highest-priority replay per HEALTH_OF_RESULTS. `regime_gated` strategy; H-ID `R-031-replay`; `predicted_effect_min: 0.5` ann-Sharpe; `min_n_trades: 200`. |

## 6. Pre-wagie historical context

For the period round-001 (2026-05-09) through round-031, the legacy `simulate_inventory_aware_sized`
harness produced a sequence of "accepted" rounds (REPORT versions before this rewrite quoted
those numbers as headline). All economic claims from that period are REVOKED pending wagie
replay. Calibration claims (round-005 per-regime ECE; round-008 Mondrian-ACI per-regime
gap closure) are KEEPABLE in principle (the calibration math is correct) but have not yet
been re-rendered under the new architecture's accept-gate format and so are not quoted as
headline here.

The legacy round-by-round narrative remains in `RESEARCH/LEDGER.md` (append-only) for
historical reference. The KILL_LIST records earlier architectural failures (rounds 010-013
under the sibling-style averaging/stacking framing).

---

## What this report intentionally OMITS

- Numerical headline tables. Until the wagie harness produces them with bootstrap CIs and
  the accept gate fires, there are no headline numbers to report. Resist the urge to
  quote pre-wagie point-estimates.
- The +20.86 ann-Sharpe round-031 finding. REVOKED. See HEALTH_OF_RESULTS.md.
- References to deleted scripts (`scripts/phase_A_round_*.py`). Those were removed in
  commit `61ce420`. Reproduction is now through `wagie experiment run <spec.yaml>`.
- The `BEFORE_vs_AFTER.csv` artefact promised by the round-017 replay roadmap. Does not
  exist. The replay has not been run.

When the first wagie-harness round accepts (bootstrap-CI on Brier improves vs baseline
with `n >= min_n_trades`), update this file's section 2 with the headline.
