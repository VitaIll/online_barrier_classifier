# online_barrier_classifier — `wagie`

> **Streaming-native ML harness for triple-barrier financial prediction. Prod ≡ backtest by construction.**

`wagie` is a small, sealed Python package for streaming ML on financial bars. The same code runs in backtest replay and live execution. Look-ahead is structurally impossible (lag bound at config time, sealed Pipeline ordering, type-checked Observation frames).

The headline metric is **calibration of the online probability `p_online`** — Brier and ECE with bootstrap CIs. Trading metrics (Sharpe, drawdown) are reported alongside but are downstream of the calibration claim.

## Architecture (post-conformal-removal)

The previous "Mondrian-ACI conformal layer" sitting on top of `p_online` has been **removed**. The streaming layer is now:

```
DataSource -> BaseBar -> Features -> Regime -> [CatBoost?] -> ARF -> LabelBuffer -> Strategy -> Risk -> Broker
                                                                |
                                                                +--> p_online (calibrated probability — system output)
```

- **CatBoost** (optional, frozen `.cbm`) emits `p_offline` as INPUT to the online layer.
- **ARF** (River `ARFClassifier`, bagged) consumes `(features, p_offline?)` and emits `p_online`.
- **Strategy** (default `ThresholdGate(p_online >= tau)`) decides when to enter.
- **No separate conformal layer.** Calibration comes from the ARF bagging + ADWIN drift surfaced by sibling STREAMING.

See [`docs/architecture.md`](docs/architecture.md) for the diagram and [`docs/concepts.md`](docs/concepts.md) for one-paragraph explanations of each component.

## Quick start (5 minutes)

```bash
git clone <repo>
cd online_barrier_classifier
pip install -e ".[dev]"
```

The committed `experiments/baseline.yaml` references `data/processed/btcusdt_1m.parquet`. A fresh clone has no parquet — generate the synthetic stand-in first:

```bash
python -c "
from pathlib import Path
import numpy as np, polars as pl
out = Path('data/processed/btcusdt_1m.parquet')
out.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42); n = 60_000; sigma = 0.0008
log_ret = rng.normal(0.0, sigma, size=n); log_ret[0] = 0.0
close = np.exp(10.0 + np.cumsum(log_ret))
rng_h = np.abs(rng.normal(0.0, sigma * 1.2, size=n))
high = close * (1.0 + rng_h); low = close * (1.0 - rng_h)
open_ = np.r_[close[0], close[:-1]]; vol = np.abs(rng.normal(100.0, 20.0, size=n))
base_ms = 1_700_000_000_000
open_time = np.arange(n) * 60_000 + base_ms; close_time = open_time + 59_999
pl.DataFrame({
    'open_time': open_time, 'open': open_, 'high': high, 'low': low,
    'close': close, 'volume': vol, 'close_time': close_time,
    'quote_volume': vol * close, 'trades': np.full(n, 50, dtype=np.int64),
    'taker_buy_base': vol * 0.5, 'taker_buy_quote': vol * close * 0.5,
    'segment_id': np.zeros(n, dtype=np.int64),
}).write_parquet(out)
print(f'wrote {out}')
"
```

Then run the baseline:

```bash
wagie experiment run experiments/baseline.yaml
```

Expected output (synthetic data; numbers are illustrative):

```
brier=0.06xxx  ece=0.0xxx  n_trades=NN  sharpe=±x.xxx
out_dir: artifacts/report
report:  artifacts/report/index.html
```

There is ONE canonical report on disk; every `wagie experiment run` overwrites
it. The previous render is zipped into `_archive/` first; only the last 10
archived snapshots are kept.

```
artifacts/report/
├── index.html          # the single unified report (open in a browser)
├── manifest.json       # what each section contributed
├── metrics.json        # MetricsBattery output (calibration leads)
├── spec.yaml           # snapshot of the spec used
├── figs/<section>/     # PNG fallbacks (when Plotly is disabled / unavailable)
├── tables/<section>.json
├── state/
│   ├── pipeline_state_hash.txt
│   └── rebuild_bundle.pkl    # fuel for `wagie report rebuild --section X`
└── _archive/<ts>_<hash>.zip  # last 10 prior renders
```

`metrics.json` always includes `brier`, `ece`, and a `trading` block (`n_trades`, `sharpe`, `max_drawdown`). Older runs that show `brier=0, ece=0` are pre-fix relics — every new run produces non-zero values whenever the labelled stream advances past warmup.

Side experiments use `--experiment NAME` and write to
`artifacts/experiments/<NAME>/` without touching the canonical report.

## CLI

```bash
wagie experiment run <spec.yaml>             # overwrite artifacts/report/
wagie experiment run <spec.yaml> --experiment foo   # side experiment, isolated
wagie experiment list                        # live manifest + archive list
wagie experiment show                        # echo metrics + report path
wagie report rebuild --section calibration   # re-render one section, no engine re-run
wagie report archives                        # list archived snapshots
wagie cv <spec.yaml>                         # cross-validation (overrides cv block)
wagie info                                   # version + public surface
```

## Make

```bash
make experiment SPEC=experiments/baseline.yaml
make cv         SPEC=experiments/baseline.yaml
make test       # full pytest
make test-fast  # contracts only
make info
```

## Adding a strategy

Strategies live in `src/wagie/strategy/__init__.py`. To add one:

1. Subclass `StrategyBase` with `@dataclass(frozen=True, slots=True)`.
2. Override `decide(self, frame, ctx) -> Sequence[Action]`. Read `frame.p_online`, `frame.in_set[alpha]`, `frame.regime_id` etc.
3. Register the kind in `STRATEGY_REGISTRY` (e.g. `"my_strategy": MyStrategy`).
4. Add a contract test in `tests/contracts/test_strategy_*.py` covering the empty-portfolio case + the active-portfolio case.
5. Reference it from a YAML spec via `wagie.strategy.kind: my_strategy`.

The repo includes `pure_conformal` (legacy; will be replaced by `threshold_gate` once sibling TRADING lands), `ev_calibrated_size`, and `regime_gated` (sibling TRADING). Do NOT write a custom training loop or chart — extend `wagie.experiments` / `wagie.charts` instead.

## Swapping the data source

`experiments/<spec>.yaml::wagie.data.parquet_path` is the single knob. Two paths:

- **Replay (backtest)**: point at any parquet matching the canonical schema (see `notebooks/01_data_download.ipynb` for synth + Binance REST sketch).
- **Live**: switch the engine to a `BinanceLiveSource`. The `Engine` accepts any `DataSource` — `ParquetReplaySource` (default) and `BinanceLiveSource` are the two ports.

Same `Pipeline`, same `Strategy`, same `Engine` event loop in both modes. Prod ≡ backtest is enforced by the `Pipeline.sealed()` construction order.

## Cross-validation

```bash
wagie cv experiments/baseline.yaml --n-folds 10 --n-test-folds 2 --embargo 5
```

Folds are purged + embargoed (D4 default = 5 decision periods). `metrics.json` includes per-fold and pooled results.

## Two-layer architecture (with offline CatBoost)

To enable the `p_offline -> p_online` two-layer flow, point `wagie.model.catboost_path` at a frozen `.cbm`:

```yaml
wagie:
  model:
    catboost_path: artifacts/offline_model/model.cbm
    selected_features_path: artifacts/offline_model/selected_features.json
```

See [`experiments/baseline_with_offline.yaml`](experiments/baseline_with_offline.yaml) for the full example. Without `catboost_path`, the ARF runs feature-only and `p_online` is its sole output (the default `experiments/baseline.yaml`).

## Tests

```bash
make test               # full suite (contracts + integration)
make test-fast          # contracts only
```

Contract tests cover algebraic laws, pipeline construction, strategy purity, replay reproducibility, position/portfolio invariants, risk-policy semantics, and the trading-console queue. Sibling TESTS adds T7 (canary) and integration in CI.

## Production readiness

Live trading is **not yet supported** out of the box. See [`docs/production_readiness.md`](docs/production_readiness.md) for the gap inventory (broker stubs, secrets, NTP, idempotency, reconciliation) and the "what would unblock paper trading" punch list.

## Repository layout

```
online_barrier_classifier/
├── src/wagie/                  THE package — overtakes all custom tooling
│   ├── core, features, pipeline, strategy, risk, io, engine
│   ├── experiments/            ExperimentProtocol — the one way to run
│   ├── training/               train(spec) — the one way to train
│   ├── metrics/                MetricsBattery — the one way to measure
│   ├── charts/                 ChartBattery — the one battery of charts
│   └── reporting/              Report — the one report
├── experiments/                user YAML specs (NOT source)
├── data/                       data — gitignored
├── artifacts/                  run outputs — gitignored
├── docs/                       concepts, architecture, production readiness
├── notebooks/                  legacy walkthroughs (see notebooks/README.md)
├── RESEARCH/                   research files — separated from software
├── agent/                      agentic-loop prompts + memory
└── tests/{contracts,integration}
```

## License

MIT.
