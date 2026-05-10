# notebooks/

> **WARNING — these notebooks are the legacy pipeline.** The canonical entry
> point is `wagie experiment run experiments/baseline.yaml`. The notebooks are
> kept for stepwise illustration and the synthetic-data generator only; do NOT
> use them as the primary research workflow. Anything published as a result must
> come from `wagie experiment run`, NOT from a notebook re-implementation.
>
> Specifically, `04_online_eval.ipynb` predates the post-conformal-removal wave
> and may still reference Mondrian-ACI / `aci.alphas` / `in_set_alpha`. The
> conformal layer has been removed; use `experiments/baseline.yaml` (default
> ARF-only) or `experiments/baseline_with_offline.yaml` (two-layer) for any new
> experiment.

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

## Running notebooks

```bash
pip install -e ".[dev]" jupyter
jupyter lab notebooks/
```

Or run cell-by-cell in VS Code's notebook UI.

**Caveats**:
- Outputs from notebooks do NOT appear in `artifacts/report/`; only `wagie experiment run`
  writes to the canonical artifact tree. A notebook run leaves nothing the
  loop discipline (per-round audit, accept gates, reproducibility hash) can
  ingest.
- Notebooks may go stale faster than the package. The package has type checks
  (pydantic `extra="forbid"`) that will raise on a renamed field; the notebooks
  may keep running with silently-wrong assumptions.
- For the synthetic-data generator (notebook 01) — that cell IS still useful;
  it's the cold-start path documented in the README quick start.

## Canonical CLI (use this, not the notebooks)

```bash
wagie experiment run experiments/baseline.yaml                # backtest mode (default)
wagie experiment run experiments/baseline_with_offline.yaml   # two-layer with CatBoost
wagie cv             experiments/baseline.yaml                # CV mode
wagie experiment list
wagie experiment show <run_id>
```
