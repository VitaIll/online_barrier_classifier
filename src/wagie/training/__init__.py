"""wagie.training — the SINGLE way to fit the offline (CatBoost) layer and
warm the streaming online layer.

Online fit is a no-op concept in this repo: ARF + ACI learn online from the
LabelBuffer inside the Engine. Offline training produces a frozen `.cbm` model
referenced from the spec's `wagie.model.catboost_path`.

Public API:
    from wagie.training import train_offline, warm_online_quantiles
    cbm_path = train_offline(spec)              # writes a .cbm
    q_init = warm_online_quantiles(spec)        # → q_init_by_regime dict

For now this re-exports the existing offline.* helpers; the goal is that
nothing else in the codebase shells out to scikit/CatBoost ad-hoc.
"""

from __future__ import annotations

from wagie.offline.label import compute_one_sided_excursion_label
from wagie.offline.warmup import warm_q_init_by_regime


def warm_online_quantiles(*args, **kwargs):
    """Compute q_init_by_regime for warm-starting the streaming Mondrian-ACI."""
    return warm_q_init_by_regime(*args, **kwargs)


def compute_labels(*args, **kwargs):
    """One-sided excursion label (the D2 contract). Single source of truth."""
    return compute_one_sided_excursion_label(*args, **kwargs)


__all__ = [
    "warm_online_quantiles", "compute_labels",
    "warm_q_init_by_regime", "compute_one_sided_excursion_label",
]
