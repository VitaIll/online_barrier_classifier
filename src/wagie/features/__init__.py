"""Feature taxonomy.

Two concrete kinds, both satisfying the unified `Feature` Protocol:
    - RiverRollingFeature   (streaming O(1)-revertable; bit-identical batch≡stream)
    - BoundedBatchFeature   (numba kernel over fixed backward window W)

All carry a FeatureSpec with `lag` and `window` bound at config time.
`forecasting_safe=True` rejects centered windows / bfill at construction.
`update_samples` exposes min history so the engine can compute warmup
automatically.
"""

from wagie.features.base import (
    Feature,
    FeatureBuilder,
    FeatureKind,
    FeatureSpec,
    FutureCovariateSpec,
    PastCovariateSpec,
)
from wagie.features.base_bar import BaseBarFeatures
from wagie.features.catalog import (
    default_streaming_catalog,
    default_streaming_features,
    make_default_regime_cuts,
)
from wagie.features.regime import RegimeCuts, RegimeFeature

__all__ = [
    "Feature",
    "FeatureBuilder",
    "FeatureKind",
    "FeatureSpec",
    "FutureCovariateSpec",
    "PastCovariateSpec",
    "BaseBarFeatures",
    "RegimeCuts",
    "RegimeFeature",
    "default_streaming_catalog",
    "default_streaming_features",
    "make_default_regime_cuts",
]
