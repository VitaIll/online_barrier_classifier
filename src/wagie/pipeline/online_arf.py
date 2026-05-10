"""Online ARF correction layer — Stage Protocol.

Reads p_offline + selected features from the Observation; emits p_online
via river.forest.ARFClassifier.

NaN feature handling (undef-flag pattern):
    Per-feature warmup gates emit NaN until window fills. The legacy short-
    circuit `if any(isnan(v)): return obs` dropped the *entire* prediction
    while ANY feature was undef'd, so with ~600 features and per-feature
    warmups the ARF stayed cold for ~max(window) bars per segment. We now
    replace each NaN with a sentinel (0.0) and emit a paired `<name>__undef`
    flag column so the ARF can learn to treat the sentinel correctly.

Drift surfacing:
    river ARF tracks per-tree ADWIN drift / warning counts (cumulative on
    `n_drifts_detected()` / `n_warnings_detected()`). We snapshot pre/post
    delta on every `transform()` and attach it to `obs.drift_signals` so the
    Engine can emit `DriftDetected` events.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

from river.forest import ARFClassifier

from wagie.core.numeric import Probability
from wagie.core.observation import Observation
from wagie.core.pipeline import StageKind


_UNDEF_SUFFIX = "__undef"
_UNDEF_SENTINEL = 0.0


class OnlineARFCorrector:
    """Adaptive Random Forest. Inputs: dict of selected_features ∪ {p_offline}.
    Output: p_online."""

    name: str = "online_arf"
    kind: StageKind = StageKind.CORRECTOR

    def __init__(
        self,
        n_models: int = 10,
        max_features: str = "sqrt",
        lambda_value: float = 6.0,
        seed: int = 42,
        selected_features: Optional[list[str]] = None,
    ):
        self.n_models = int(n_models)
        self.max_features = max_features
        self.lambda_value = float(lambda_value)
        self.seed = int(seed)
        self.selected_features = list(selected_features) if selected_features else None
        self._arf: Optional[ARFClassifier] = None
        self._n_seen = 0
        # Cumulative drift counters (mirrored from the river ARF since cold-start;
        # `state_dict` exposes these for monitoring / dashboards).
        self._n_drifts_total = 0
        self._n_warnings_total = 0

    def _ensure_init(self) -> None:
        if self._arf is None:
            self._arf = ARFClassifier(
                n_models=self.n_models,
                max_features=self.max_features,
                lambda_value=self.lambda_value,
                seed=self.seed,
            )

    @staticmethod
    def _is_nan(v) -> bool:
        return isinstance(v, float) and math.isnan(v)

    def _z(self, obs: Observation) -> dict:
        """Build the raw feature dict (may contain NaNs)."""
        bar_dict = obs.to_dict()
        if self.selected_features is None:
            # Pass everything except metadata/decision/etc.
            return {k: v for k, v in bar_dict.items()
                    if not k.startswith("_") and k not in {"decision"}}
        z = {f: bar_dict.get(f, float("nan")) for f in self.selected_features}
        if obs.p_offline is not None:
            z["p_offline"] = float(obs.p_offline)
        return z

    def _z_with_undef_flags(self, obs: Observation) -> dict:
        """Build the ARF feature dict using the undef-flag pattern.

        For every feature key, emit:
            - the value (sentinel 0.0 if NaN/missing)
            - a paired `<name>__undef` flag (1 if NaN/missing, else 0)

        The flag lets the ARF learn that the sentinel is structurally distinct
        from a true zero. This eliminates the cold-start NaN drop bug."""
        z_raw = self._z(obs)
        out: dict = {}
        for k, v in z_raw.items():
            if v is None or self._is_nan(v):
                out[k] = _UNDEF_SENTINEL
                out[f"{k}{_UNDEF_SUFFIX}"] = 1
            else:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = _UNDEF_SENTINEL
                    out[f"{k}{_UNDEF_SUFFIX}"] = 1
                    continue
                out[f"{k}{_UNDEF_SUFFIX}"] = 0
        return out

    def _snapshot_drift(self) -> tuple[int, int]:
        """Return (n_drifts_total, n_warnings_total) from the underlying ARF.

        river exposes `n_drifts_detected()` / `n_warnings_detected()` as
        cumulative methods (sum over per-tree counters). We snapshot to
        compute per-iteration deltas in `transform`."""
        if self._arf is None:
            return 0, 0
        n_d = 0
        n_w = 0
        try:
            n_d = int(self._arf.n_drifts_detected())
        except Exception:
            try:
                tracker = getattr(self._arf, "_drift_tracker", None)
                n_d = int(sum(tracker.values())) if tracker else 0
            except Exception:
                n_d = 0
        try:
            n_w = int(self._arf.n_warnings_detected())
        except Exception:
            try:
                tracker = getattr(self._arf, "_warning_tracker", None)
                n_w = int(sum(tracker.values())) if tracker else 0
            except Exception:
                n_w = 0
        return n_d, n_w

    def transform(self, obs: Observation) -> Observation:
        self._ensure_init()
        self._n_seen += 1
        # Undef-flag pattern: NaN no longer short-circuits the prediction.
        z = self._z_with_undef_flags(obs)
        try:
            proba = self._arf.predict_proba_one(z)
        except Exception:
            return obs
        p1 = float(proba.get(1, 0.5)) if proba else 0.5
        out = obs.with_p_online(Probability(min(max(p1, 0.0), 1.0)))
        # Surface drift signals via cumulative snapshot — `transform` itself
        # does not move drift counters (only `learn_one` does), so we expose
        # the *cumulative totals* here. Callers that want per-iteration deltas
        # should diff these between successive bars.
        n_d, n_w = self._snapshot_drift()
        out = out.with_drift_signals({
            "n_drifts_total": int(n_d),
            "n_warnings_total": int(n_w),
        })
        return out

    def update(self, obs: Observation, label: Optional[int] = None) -> None:
        if label is None:
            return
        self._ensure_init()
        # Snapshot pre-learn drift counters; ADWIN fires inside learn_one.
        pre_d, pre_w = self._snapshot_drift()
        z = self._z_with_undef_flags(obs)
        try:
            self._arf.learn_one(z, int(label))
        except Exception:
            return
        post_d, post_w = self._snapshot_drift()
        if post_d > pre_d:
            self._n_drifts_total = post_d
        if post_w > pre_w:
            self._n_warnings_total = post_w

    # ---- Stateful ----

    def state_dict(self) -> dict:
        n_d, n_w = self._snapshot_drift()
        return {
            "n_seen": self._n_seen,
            "seed": self.seed,
            "n_models": self.n_models,
            "lambda_value": self.lambda_value,
            "n_drifts": int(n_d),
            "n_warnings": int(n_w),
        }

    def load_state_dict(self, state: dict) -> None:
        self._n_seen = int(state.get("n_seen", 0))
        self._n_drifts_total = int(state.get("n_drifts", 0))
        self._n_warnings_total = int(state.get("n_warnings", 0))

    def state_hash(self) -> bytes:
        h = hashlib.sha256()
        h.update(b"ARF|")
        h.update(repr((self.n_models, self.lambda_value, self.seed,
                       self.max_features, self._n_seen)).encode())
        return h.digest()

    def reset(self) -> None:
        self._arf = None
        self._n_seen = 0
        self._n_drifts_total = 0
        self._n_warnings_total = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def n_drifts(self) -> int:
        n_d, _ = self._snapshot_drift()
        return int(n_d)

    @property
    def n_warnings(self) -> int:
        _, n_w = self._snapshot_drift()
        return int(n_w)


__all__ = ["OnlineARFCorrector"]
