"""Offline tools — used to fit p_offline once and never again at runtime.

label.py:    one-sided excursion label generator (D2). Per the polars-compat
             verdict we DO NOT call mlfinpy here; we implement the simple
             one-bar-ahead label natively in polars (the existing
             feature_build.ipynb logic, preserved 1:1 per D6).
train.py:    skrub.TableVectorizer + CatBoost fit; outputs a frozen .cbm.
cv.py:       skfolio.CombinatorialPurgedCV with embargo=5 decision periods (D4).
warmup.py:   warm q_init_by_regime for the streaming Mondrian-ACI calibrator.
"""

from wagie.offline.label import compute_one_sided_excursion_label
from wagie.offline.warmup import warm_q_init_by_regime

__all__ = ["compute_one_sided_excursion_label", "warm_q_init_by_regime"]
