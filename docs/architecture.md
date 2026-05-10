# Architecture (post-conformal-removal)

One-pager. The Mondrian-ACI conformal layer that previously sat on top of `p_online` has been **removed**. Calibration is now provided by the ARF bagging + ADWIN drift surfacing alone.

```
+-------------------+         +---------------+         +---------------+
| ParquetReplay     |         | BinanceLive   |         | (any other    |
| Source (default)  |         | Source        |         |  DataSource)  |
+---------+---------+         +-------+-------+         +-------+-------+
          |                           |                         |
          +---------------+-----------+-------------------------+
                          |
                          v
                   +------+------+
                   |   Engine    |  event loop, single source of truth
                   +------+------+
                          |
                          v
+-------------------------------------------------------------+
|                     Pipeline (sealed)                       |
|                                                             |
|  BaseBar -> Features -> Regime -> [CatBoost?] -> ARF        |
|                                          |         |        |
|                                          |         v        |
|                                          |     p_online --+ |
|                                          v                | |
|                                     p_offline             | |
|                                  (optional input)         | |
|                                                           v |
|                                                    LabelBuffer  (delayed-label rule)
|                                                           |
|                                                           v
|                                                       Strategy
|                                                       (ThresholdGate p_online >= tau,
|                                                        RegimeGated, ...)
|                                                           |
+-----------------------------------------------------------+
                          |
                          v
                   +------+------+
                   | RiskEngine  |  policies: max-positions, max-DD, max-loss-per-pos,
                   |             |            max-order-rate, kill-switch
                   +------+------+
                          |
                          v
                   +------+------+
                   |  SimBroker  |  (replay) | BinanceBroker (live)
                   +------+------+
                          |
                          v
                   +------+------+
                   | Portfolio   |  realized + open positions, equity, drawdown
                   +------+------+
                          |
                          v
            +-------------+-------------+
            |                                                                |
            v                                                                v
+-----------+--------+                                       +----------+----+
| MetricsBattery     |                                       | ChartBattery   |
| brier, ece,        |                                       | reliability,   |
| per-regime,        |                                       | equity, regime |
| trading{n,sharpe}  |                                       | scatter, ...   |
+-----------+--------+                                       +----------+----+
            |                                                                |
            +--------------------------+--------------------------------------+
                                       |
                                       v
                              +--------+--------+
                              |   Report (md)   |
                              | per-run dashboard|
                              +-----------------+
```

## Stage ordering (sealed)

The `Pipeline` constructor enforces monotone `StageKind` ordering at build time:

```
BASE_BAR < FEATURES < REGIME < OFFLINE_PREDICTOR < ONLINE_CORRECTOR
        < LABEL_BUFFER < STRATEGY
```

Sibling INVARIANTS adds `STAGE_REGISTRY` so any stage out of order raises at construction (not at run time).

## Two-layer flow (with offline)

```
20m bars -> Features -> CatBoost (frozen) -> p_offline
                                |
                                v
                            ARF (online) consumes (features, p_offline)
                                |
                                v
                            p_online (calibrated probability — system output)
                                |
                                v
                            ThresholdGate(p_online >= tau)
```

Without `wagie.model.catboost_path` set, the CatBoost stage is bypassed and the ARF runs feature-only.

## What changed (vs the "two-layer + conformal" architecture)

- **Removed**: `MondrianACICalibrator` stage. The Pipeline previously emitted `q_lo[alpha]` and `in_set[alpha]` per bar, and `PureConformalGate` opened on `in_set[alpha] == 1`. That layer is gone.
- **Replaced**: gating is now `ThresholdGate(p_online >= tau)`. `tau` is chosen on a validation Sharpe (or other configurable) sweep.
- **Why**: sibling ARCH owns the cleanup. The conformal layer was achieving its design goal (regime-conditional coverage, `≤ 0.6pp gap` per round-008) but the coverage signal was not translating into tradable edge under the current label / cost / barrier-strategy combination. The plain probability threshold on the bagging-calibrated `p_online` is simpler, equivalently predictive at the chosen `tau`, and removes one layer of state to reason about.

## What did NOT change

- The two-layer flow (`p_offline` -> `p_online` is still the architecture when CatBoost is wired in).
- The label, the regime cuts, the streaming feature extractors, the LabelBuffer (delayed-label rule), the Engine event loop.
- Reporting: calibration leads. Brier / ECE / per-regime calibration curves before any trading metric.

See [`concepts.md`](concepts.md) for one-paragraph explanations of each component and [`production_readiness.md`](production_readiness.md) for the gap inventory between the current state and live execution.
