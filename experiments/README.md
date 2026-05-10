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

## Available specs

| Spec | Purpose |
|---|---|
| `baseline.yaml` | Default — ARF only, no offline. Cold-start the synth parquet first (see README quick start). |
| `baseline_with_offline.yaml` | Two-layer: CatBoost -> ARF -> ThresholdGate. Requires a frozen `.cbm` at `artifacts/offline_model/model.cbm`. |
| `sweep_alpha.yaml` | Five-spec sweep over `wagie.strategy.tau` (the new gating knob). Multi-doc YAML. |
| `replay_r031_low_vol_gate.yaml` | Highest-priority replay per HEALTH_OF_RESULTS — `regime_gated` strategy with `R-031-replay` H-ID and accept-gate fields. |

## Spec schema (post-conformal-removal)

A spec has these blocks (all `extra: forbid` — typos are errors). The
post-conformal-removal wave removed the `wagie.model.aci` block and the
`pure_conformal` strategy; sibling RIGOR added the top-level accept-gate
fields (`hypothesis_id`, `predicted_effect_min`, `min_n_trades`, `bootstrap`);
sibling TRADING added `threshold_gate` and `regime_gated` strategy kinds plus
the richer `broker` cost model.

```yaml
name: baseline                                # short identifier
description: "..."
seed: 42

# RIGOR — accept-gate fields (top-level; outside the wagie block)
hypothesis_id: BASELINE-arf-only              # H-ID this round tests
predicted_effect_min: 0.0                     # CI lower bound the headline must beat to accept
min_n_trades: 50                              # below this n, the round is statistically inadmissible
bootstrap:
  scheme: stationary_block                    # stationary_block | iid | none
  n_resamples: 5000
  block_length: auto                          # Politis-White plug-in

wagie:                                        # WagieConfig (engine config)
  data:
    parquet_path: data/processed/btcusdt_1m.parquet
    m_minutes: 20
  model:
    catboost_path: null                       # offline model (optional)
    selected_features_path: null              # frozen feature column order
    arf:    {n_models: 10, lambda_value: 6.0, seed: 42}
    # NO `aci:` block — the conformal layer has been removed.
  strategy:
    # TRADING ships these kinds: threshold_gate, regime_gated, ev_calibrated_size
    kind: threshold_gate
    layer: p_online                           # p_online | p_offline (which probability to gate on)
    tau: 0.30                                 # the gating threshold
    extra: {}                                 # for regime_gated: regime_signal, keep_regimes, ...
  broker:
    inventory_cap: 5
    take_profit_log: 0.0041113
    stop_loss_log: 0.0041113                  # set to a large value (e.g. 1.0e9) for c_stop=inf
    cost_bps: 1.0                             # TRADING's richer cost model: maker/taker, slippage
    execution_latency_minutes: 1              # honest execution timing (D3)
    expiry_minutes: 20
  runtime:
    warmup_samples: 96

features:
  catalog: default                            # default | minimal | full
  n_features: null                            # cap (or null = use all)
  regime_feature: parkinson_var_rolling_mean_24
  regime_edges:  [1.0e-6, 1.0e-5]
  regime_labels: [low, med, high]

charts:   {enable: true, n_calibration_bins: 10}
report:   {enable: true, title: "baseline experiment"}
artifacts: {out_dir: artifacts/runs, save_state: true, save_predictions: true}
```

## Conventions

- One spec = one named experiment. Vary `name` per spec (artifact dirs collide otherwise).
- Never bypass the protocol. If you find yourself writing a custom training
  loop or chart, extend `wagie.experiments` / `wagie.charts` instead.
- Specs are committed; runs (`artifacts/`) are not.
- Replay specs MUST set `hypothesis_id` to the round being replayed (e.g. `R-031-replay`)
  and a non-zero `predicted_effect_min`. A replay that "passes" without the
  bootstrap-CI lower bound exceeding the prediction does NOT accept the original
  round; it merely confirms the harness runs.
