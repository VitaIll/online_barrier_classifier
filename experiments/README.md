# experiments/

User-defined experiment **specs**. This directory contains only YAML — no Python.
Each YAML file describes one experiment via `wagie.experiments.ExperimentSpec`.

## Run an experiment

```bash
wagie experiment run experiments/baseline.yaml
# or:
python -m wagie experiment run experiments/baseline.yaml
```

Outputs go to `artifacts/runs/<run_id>/` (gitignored): `spec.yaml`,
`metrics.json`, `charts/`, `report.md`, `state/`.

## Browse runs

```bash
wagie experiment list                       # one line per run
wagie experiment show 20260510-103040_baseline_a1b2c3d4
```

## Spec schema

A spec has these blocks (all `extra: forbid` — typos are errors):

```yaml
name: baseline                              # short identifier
description: "..."
seed: 42

wagie:                                      # WagieConfig (engine config)
  data:
    parquet_path: data/processed/btcusdt_1m.parquet
    m_minutes: 20
  model:
    catboost_path: null                     # offline model (optional)
    arf:    {n_models: 10, lambda_value: 6.0}
    aci:    {alphas: [0.05, 0.10, 0.20], gamma: 0.01}
  strategy:
    kind: pure_conformal
    alpha: 0.10
  broker:
    inventory_cap: 5
  runtime:
    warmup_samples: 96

features:
  catalog: default                          # default | minimal | full
  n_features: null                          # cap (or null = use all)
  regime_feature: parkinson_var_rolling_mean_24
  regime_edges:  [1.0e-6, 1.0e-5]
  regime_labels: [low, med, high]

charts:   {enable: true, n_calibration_bins: 10}
report:   {enable: true, title: "baseline experiment"}
artifacts: {out_dir: artifacts/runs, save_state: true, save_predictions: true}
```

## Conventions

- One spec = one named experiment. Vary `name` per spec.
- Never bypass the protocol. If you find yourself writing a custom training
  loop or chart, extend `wagie.experiments`/`wagie.charts` instead.
- Specs are committed; runs (`artifacts/`) are not.
