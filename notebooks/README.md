# notebooks/

Walk-through of the full pipeline using the `wagie` package. Each notebook is
small, runnable cold against a synthetic parquet, and **uses package APIs
exclusively** — no bespoke training loops, no ad-hoc charts. They are
demonstrations, not extensions.

| # | Notebook | Calls into |
|---|---|---|
| 01 | [data_download](01_data_download.ipynb) | synthetic generator + Binance REST sketch (no package dep) |
| 02 | [feature_build](02_feature_build.ipynb) | `wagie.training.compute_labels`, `wagie.training.warm_online_quantiles` |
| 03 | [offline_train](03_offline_train.ipynb) | `catboost`, `wagie.metrics.{brier_score, expected_calibration_error, roc_auc, pr_auc}` |
| 04 | [online_eval](04_online_eval.ipynb) | `wagie.experiments.{ExperimentSpec, ExperimentProtocol, CVSpec}` |

Same flow as the four pre-rewire notebooks (`data_download`, `feature_build`,
`offline_train`, `online_eval`), but every step now goes through the package.

## Run

```bash
pip install -e ".[dev]" jupyter
jupyter lab notebooks/
```

Or run cell-by-cell in VS Code's notebook UI.

## Equivalent CLI

The 04 notebook is a hands-on equivalent of:

```bash
wagie experiment run experiments/baseline.yaml      # backtest mode
wagie cv             experiments/baseline.yaml      # CV mode
wagie experiment list
wagie experiment show <run_id>
```
