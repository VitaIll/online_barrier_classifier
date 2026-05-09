# online_barrier_classifier — `wagie`

> **Streaming-native ML harness for triple-barrier financial prediction. Prod ≡ backtest by construction.**

`wagie` is a small, sealed Python package for streaming ML on financial bars. Same code runs in backtest replay and live execution. Look-ahead is structurally impossible (lag bound at config time, sealed Pipeline ordering, type-checked Observation frames).

## One protocol, one entry point

There is exactly **one** way to run an experiment in this repo:

```bash
wagie experiment run experiments/baseline.yaml
```

The protocol writes everything to `artifacts/runs/<run_id>/`:

```
artifacts/runs/20260510-103040_baseline_a1b2c3d4/
├── spec.yaml          # snapshot of the spec used
├── metrics.json       # MetricsBattery output (calibration leads)
├── charts/            # ChartBattery output (PNGs)
├── report.md          # Report renderer output
└── state/             # pipeline state hash
```

No bespoke scripts. Custom training loops, ad-hoc charting, one-off reports — **forbidden**. Extend the package instead.

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
├── RESEARCH/                   research files — separated from software
├── agent/                      agentic-loop prompts + memory
└── tests/{contracts,integration}
```

## CLI

```bash
wagie experiment run <spec.yaml>     # run a single experiment
wagie experiment list                 # one line per run in artifacts/runs/
wagie experiment show <run_id>        # echo metrics + report path
wagie cv <spec.yaml>                  # cross-validation
wagie info                            # version + public surface
```

## Make

```bash
make experiment SPEC=experiments/baseline.yaml
make cv         SPEC=experiments/baseline.yaml
make test       # full pytest
make test-fast  # contracts only
make info
```

## Install

```bash
git clone <repo>
cd online_barrier_classifier
pip install -e ".[dev]"
```

## Design

Engine event loop, single source of truth:

```
DataSource → Engine → Pipeline (BaseBar | Features | Regime | CatBoost? | ARF | ACI | LabelBuffer | Strategy)
                │
                ├─ Strategy.decide(obs, ctx) → Sequence[Action]
                ├─ RiskEngine.check(action, portfolio) → approved/rejected
                ├─ SimBroker.dispatch(action) → Position lifecycle events
                └─ ExperimentProtocol → MetricsBattery → ChartBattery → Report
```

Calibration leads. The online layer's value shows in Brier/ECE, not ranking metrics. Every `wagie experiment run` produces a reliability diagram before anything else.

## Tests

```bash
make test               # full suite (contracts + integration)
make test-fast          # contracts only (75 tests, <90s)
```

Contract tests cover algebraic laws, pipeline construction, strategy purity, replay reproducibility, position/portfolio invariants, risk-policy semantics, and the trading-console queue.

## License

MIT.
