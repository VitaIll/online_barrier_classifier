# Pending tests — re-enable as utilities are built

These tests were ported from the sibling `barrier_classifier` project. They
test functions that exist in that project's `src/utils.py` (~5000 lines)
but **not** in this project's minimal `src/utils.py`. They are excluded
from `pytest` (see `pyproject.toml::pytest.norecursedirs`).

Restore tests as the autonomous loop's rounds add the corresponding utilities
to `src/utils.py`:

| Test file | Functions it requires | Likely owning round |
|---|---|---|
| `test_weights.py` | `compute_barrier_distance_weight`, `compute_time_discount_weight`, `compute_training_weights` | H-020 / H-102 (port sample weighting) |
| `test_properties.py` | weighting + `deflated_sharpe` + `get_imputation_value` | After H-102 + H-103 land |
| `test_features.py` | `compute_base_series`, `get_imputation_value`, `create_undef_flags_and_impute` | H-103 (port feature pipeline scaffolding) |

Procedure to re-enable a single file:
1. The owning round adds the missing function(s) to `src/utils.py` (or wherever they belong) **with adaptations** for this project's online + offline architecture (M=20, online ARF correction layer present, etc.).
2. Move the test back to `tests/`.
3. Run `make test` to confirm green.
4. Commit on master per `RESEARCH/LOOP_DISCIPLINE.md`.

## Closed parks

- `test_causality.py` and `test_splits.py` (originally parked under H-101) were
  removed in round-003 work for H-101: their sibling-imported semantics
  (`chronological_split_with_embargo`, `walk_forward_cv`,
  `compute_past_target_features`, minute-bar `construct_labels`) do not match
  this project's contract. The replacement is `tests/test_label_split_utils.py`
  which pins THIS project's semantics directly:
  - `compute_log_excursion`, `calibrate_alpha`, `construct_labels` for the
    next-bar-high label.
  - `chronological_split` (no embargo, no walk-forward CV).
  - Round-trip against the persisted `bars_20m_features.parquet` — skipped
    when data is not on disk.
