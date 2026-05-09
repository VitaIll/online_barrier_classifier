"""CatBoost ensemble wrapper — averaged predictions across seed-varied models.

Ported from the sibling `barrier_classifier` (src/utils.py::CatBoostEnsemble) with
two additions tailored to this project:

1. `load_ensemble(base_path, n_models)` classmethod: round-trips with `save_model`.
   The sibling has save but not load — load is needed for the autonomous loop's
   model-reload pattern (offline notebook persists, downstream notebooks reload).
2. `n_models` property and `__len__` for symmetry with `len(ensemble)`.

The class is a thin wrapper: it takes a list of *already-fitted* CatBoost models,
averages their probabilistic predictions / feature importances / best iterations,
and saves them cleanly under `<base>` + `<base>.{i}.cbm`. Fitting is the caller's
responsibility — that's where the seed variation lives.

Usage:
    models = [CatBoostClassifier(random_seed=s, ...).fit(X, y) for s in seeds]
    ens = CatBoostEnsemble(models)
    p = ens.predict_proba(X_test)        # averaged over seeds
    ens.save_model("artifacts/offline_model/model.cbm")
    # later:
    ens2 = CatBoostEnsemble.load_ensemble("artifacts/offline_model/model.cbm",
                                           n_models=len(models))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class CatBoostEnsemble:
    def __init__(self, models: list[Any]):
        if not models:
            raise ValueError("CatBoostEnsemble requires at least one model.")
        self.models = list(models)

    def __len__(self) -> int:
        return len(self.models)

    @property
    def n_models(self) -> int:
        return len(self.models)

    def predict_proba(self, X) -> np.ndarray:
        probas = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return probas.mean(axis=0)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            return (p[:, 1] >= threshold).astype(int)
        return (p >= threshold).astype(int)

    def get_feature_importance(self, data=None, type: str = "PredictionValuesChange") -> np.ndarray:
        imps = [
            np.asarray(m.get_feature_importance(data=data, type=type), dtype=float)
            for m in self.models
        ]
        return np.mean(imps, axis=0)

    def get_best_iteration(self) -> int:
        return int(np.mean([m.get_best_iteration() for m in self.models]))

    def get_evals_result(self) -> dict[str, dict[str, list[float]]]:
        results = [m.get_evals_result() for m in self.models]
        averaged: dict[str, dict[str, list[float]]] = {}
        for split_name, metrics in results[0].items():
            averaged[split_name] = {}
            for metric_name in metrics:
                curves = [r[split_name][metric_name] for r in results]
                min_len = min(len(c) for c in curves)
                if min_len == 0:
                    averaged[split_name][metric_name] = []
                    continue
                truncated = [np.asarray(c[:min_len], dtype=float) for c in curves]
                averaged[split_name][metric_name] = np.mean(truncated, axis=0).tolist()
        return averaged

    def save_model(self, path: str | Path) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.models):
            m.save_model(str(base.with_suffix(f".{i}.cbm")))
        self.models[0].save_model(str(base))

    @classmethod
    def load_ensemble(
        cls,
        base_path: str | Path,
        n_models: int,
        model_class: type | None = None,
    ) -> "CatBoostEnsemble":
        if n_models < 1:
            raise ValueError(f"n_models must be >= 1, got {n_models}")
        base = Path(base_path)
        if model_class is None:
            from catboost import CatBoostClassifier
            model_class = CatBoostClassifier
        models = []
        for i in range(n_models):
            shard = base.with_suffix(f".{i}.cbm")
            if not shard.exists():
                raise FileNotFoundError(
                    f"shard {i} missing: {shard}. "
                    f"Expected {n_models} shards under base={base}."
                )
            m = model_class()
            m.load_model(str(shard))
            models.append(m)
        return cls(models)
