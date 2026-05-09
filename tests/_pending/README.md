# Pending tests — re-enable as utilities are built

These tests were ported from the sibling `barrier_classifier` project. They
test functions that exist in that project's `src/utils.py` (~5000 lines)
but **not** in this project's minimal `src/utils.py`. They are excluded
from `pytest` (see `pyproject.toml::pytest.norecursedirs`).

Restore tests as the autonomous loop's rounds add the corresponding utilities
to `src/utils.py`:

| Test file | Functions it requires | Likely owning round |
|---|---|---|
| `test_causality.py` | `construct_labels`, `chronological_split_with_embargo`, `walk_forward_cv`, `compute_past_target_features` | H-101 (port label + split utilities) |
| `test_weights.py` | `compute_barrier_distance_weight`, `compute_time_discount_weight`, `compute_training_weights` | H-020 / H-102 (port sample weighting) |
| `test_splits.py` | `chronological_split_with_embargo`, `walk_forward_cv` | H-101 |
| `test_properties.py` | All of the above + `get_imputation_value`, `deflated_sharpe` | After H-101 + H-102 land |
| `test_features.py` | `compute_base_series`, `get_imputation_value`, `create_undef_flags_and_impute` | H-103 (port feature pipeline scaffolding) |

Procedure to re-enable a single file:
1. The owning round adds the missing function(s) to `src/utils.py` (or wherever they belong) **with adaptations** for this project's online + offline architecture (M=20, online ARF correction layer present, etc.).
2. Move the test back to `tests/`.
3. Run `make test` to confirm green.
4. Commit on the round branch.
