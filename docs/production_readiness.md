# Production readiness

What's missing for live trading. Honest gap inventory; no aspiration creep.

## TL;DR

`wagie` runs cleanly in **backtest replay mode**. Live trading is **not yet supported** out of the box. The contract `prod ≡ backtest` holds at the **Pipeline** level (same code path, same Stage ordering); it does NOT yet hold end-to-end (live broker, secrets, observability, NTP, idempotency, reconciliation are stubs or missing).

This page enumerates exactly what's missing and exactly what would unblock paper trading.

---

## Gap inventory

### 1. Broker integration (live)

- **What exists**: `wagie.io.brokers.SimBroker` (replay) is fully wired. `BinanceBroker` is a placeholder/stub — does NOT actually place orders.
- **Missing**:
  - Real REST + WebSocket session management (auth, heartbeat, reconnect with state).
  - Order lifecycle handling end-to-end: `NEW -> PARTIALLY_FILLED -> FILLED | CANCELED | REJECTED | EXPIRED`.
  - Reduce-only / IOC / FOK semantics matched to the venue's actual flags.
  - Margin / isolated-margin mode awareness (Binance Futures vs Spot).
  - Funding-rate accrual on perpetual futures.
- **Risk if shipped as-is**: orders silently dropped or duplicated; no recovery.

### 2. Secrets management

- **What exists**: nothing. `BinanceBroker` has no auth at all.
- **Missing**:
  - API-key + secret loaded from a secrets vault (env var as fallback ONLY for dev).
  - Per-environment isolation (paper / staging / prod with distinct keys).
  - Read-only key for monitoring, write key for execution, separated.
  - Secret rotation procedure documented (rotate without restart, or graceful drain).
- **Risk if shipped as-is**: credentials in source / logs.

### 3. Observability

- **What exists**: structured-ish Python logging at INFO. `metrics.json` per run.
- **Missing**:
  - Prometheus metrics endpoint with `/metrics` exposing Pipeline lag, ARF predict latency, broker round-trip, queue depths, drift (ADWIN signal count).
  - Distributed tracing (OpenTelemetry) on the Engine event loop — one trace per decision bar end-to-end.
  - Alert rules: ARF stops emitting, broker disconnect, PnL < threshold, drift count spike.
  - Live calibration drift dashboard (Brier / ECE rolling window).
- **Risk if shipped as-is**: silent failure; no SLO enforceable.

### 4. NTP / clock sync

- **What exists**: `wagie.io.clock.LiveClock` reads system time.
- **Missing**:
  - Hard precondition that NTP is synchronized within 50 ms; a degraded-clock guard that pauses trading if drift exceeds the threshold.
  - Timestamp source recorded with every Order (system-clock, exchange-ack-clock).
  - Skew monitoring per minute; alert on > 100 ms drift.
- **Risk if shipped as-is**: misordered events vs the venue's view of time; cannot reproduce live runs.

### 5. Idempotency

- **What exists**: nothing.
- **Missing**:
  - Per-Action idempotency key (UUID + decision-bar-index) propagated to the broker as `clientOrderId`.
  - On reconnect / replay, never resubmit an Action whose `clientOrderId` is already known to the venue.
  - Persistence of in-flight Actions across process restart.
- **Risk if shipped as-is**: duplicate orders on transient network failure.

### 6. Reconciliation

- **What exists**: nothing.
- **Missing**:
  - Periodic poll of venue-side open positions + balance, diffed against `Portfolio` state; alert on any mismatch.
  - End-of-session reconciliation report: realized PnL (local) vs realized PnL (venue), filled vs expected.
  - Crash-recovery: on restart, resume `Portfolio` state from venue + last-known checkpoint, NOT from a stale in-memory copy.
- **Risk if shipped as-is**: phantom positions or unaccounted PnL after any restart or partial fill.

### 7. State checkpointing

- **What exists**: `artifacts/runs/<run_id>/state/` (per-run pipeline state hash).
- **Missing**:
  - **Continuous** checkpointing during a live session (not only at run end).
  - Atomic write + rename so a crash mid-checkpoint doesn't corrupt state.
  - Replay-from-checkpoint smoke test in CI.
- **Sibling INVARIANTS** is closing this gap with structured logging + state checkpointing.

### 8. Risk policy enforcement (live mode)

- **What exists**: `RiskEngine` with `MaxPositionsPolicy`, `MaxDrawdownPolicy`, `MaxLossPerPositionPolicy`, `MaxOrderRatePolicy`, `KillSwitchPolicy`.
- **Missing**:
  - Live kill-switch wired to a real "halt all trading" venue API (cancel-all-orders with confirmation), not just the in-process flag.
  - Drawdown computed against a venue-truth balance, not `Portfolio.realized_pnl_log`.
  - Manual override from a side-channel (Slack command, signed CLI) tested in staging.

### 9. Data-source robustness (live)

- **What exists**: `BinanceLiveSource` outline. `ParquetReplaySource` is fully working.
- **Missing**:
  - Gap-handling (kline missed minute) — currently the Pipeline assumes a contiguous 1m stream.
  - Late/duplicate kline detection.
  - Backfill on reconnect: pull missed minutes via REST and replay through the Pipeline before resuming live.
  - Schema validation per kline (defensive — Binance has changed kline schemas mid-year before).

### 10. Calibration / drift surfacing (online-mode)

- **What exists**: ARF + ADWIN per-tree drift detector internal to the ARF.
- **Missing**:
  - Top-level ADWIN on the LAC score `s_t = 1 - p_online(y_t | x_t)` (sibling STREAMING is adding this).
  - Auto-trigger: on detected drift, log a `DRIFT` event with the rolling Brier delta + force-pause for human review (no auto-retrain in v1 — human-in-the-loop only).
  - Per-regime drift monitoring (regime cuts are frozen on train; live regime-membership shift by itself indicates drift).

---

## What would unblock paper trading

Shortest path from current state to "I can let this run on testnet for a week and trust the result":

1. **Secrets**: env-var loader + a `paper-trading.local.yaml` template excluded from git. (~1 day)
2. **BinanceBroker**: implement `place_order` + `cancel_order` against Binance Futures **testnet** (NOT mainnet); subscribe to user-data WebSocket for fill events; map fills to `Position` lifecycle. (~3 days)
3. **Idempotency**: thread a `clientOrderId` through every `Action` from `Strategy.decide` to `BinanceBroker.dispatch`; persist `in_flight_actions.json` next to the run. (~1 day)
4. **Reconciliation**: at start of session pull venue positions + balance and either match `Portfolio` or refuse to start. End of session, diff and emit reconciliation report. (~2 days)
5. **Kill-switch wiring**: connect `KillSwitchPolicy.trip()` to `BinanceBroker.cancel_all_orders()` + halt the Engine event loop. Manual side-channel deferred to v2. (~1 day)
6. **Continuous checkpointing**: write `state/checkpoint.parquet` every N decision bars, atomic rename, replay-from-checkpoint test. (~1 day)
7. **NTP guard**: refuse to start the Engine if `abs(system_time - venue_time) > 50ms` at startup; warn if it drifts during the session. (~half a day)
8. **Observability minimum**: Prometheus `/metrics` endpoint exposing decision-bar count, broker round-trip ms, ARF predict ms, drift events. Grafana dashboard JSON checked in. (~2 days)
9. **Live data robustness**: gap detection + backfill from REST on reconnect. (~2 days)
10. **Drift surfacing**: top-level ADWIN on LAC score, drift-event logging. **Sibling STREAMING is shipping this.**

**Total realistic estimate**: 2–3 weeks of focused engineering (one engineer) to get to "trustworthy testnet". Mainnet adds: post-mortems on testnet runs, a real risk policy review, a manual kill-switch drill, and a smaller-than-default position sizing for the first month.

## What is explicitly NOT in scope (v1 paper trading)

- Auto-retrain on drift. Drift surfacing only; human review required.
- Multi-venue routing.
- Cross-instrument risk netting.
- Tax-lot accounting.
- Regulatory reporting (best-execution, transaction reports).

These are v2+ items. None of them block testnet paper trading.

---

## See also

- [`concepts.md`](concepts.md) — what each Pipeline stage does.
- [`architecture.md`](architecture.md) — the post-conformal-removal architecture diagram.
- [`../RESEARCH/CONSTITUTION.md`](../RESEARCH/CONSTITUTION.md) — invariants the autonomous loop must obey (causality, prequential discipline, strategy-realism).
