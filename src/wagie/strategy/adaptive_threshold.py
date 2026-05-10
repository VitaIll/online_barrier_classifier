"""AdaptiveThresholdController — P-controlled τ with EWMA tracking + pause hysteresis.

Closes a feedback loop on a probability gate:

    r_hat_ewma   = EWMA of "did we enter" (entered ∈ {0, 1})
    r_star_ewma  = EWMA of "did the matured label fire" (label ∈ {0, 1})
    err          = r_hat_ewma - r_star_ewma
    Δτ           = clip(γ · err, ±dtau_max)
    τ           ← clip(τ + Δτ, [tau_floor, tau_ceil])

Intuition: when we're entering more often than the realized barrier-hit rate
(predictor over-confident), τ rises and tightens entries. When the predictor
is under-confident (we'd hit the barrier more than we open), τ falls.

Pause hysteresis: when r_star_ewma drops below r_min the controller pauses
(no new entries); resumes once r_star_ewma climbs back above r_resume. Updates
to the pause flag only fire on bars with a fresh matured label — the entered
EWMA alone never flips the gate.

Default ``lambda_decay`` of 2e-3 corresponds to ≈ 350-bar half-life — about
4 days for the project's 20-minute bars. Override at construction for other
bar cadences.

This is a pure dataclass; no I/O, no random state. The Strategy that owns it
is responsible for calling ``observe`` once per bar (with the previous bar's
entered flag and any newly-matured label) and ``should_enter`` to gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdaptiveThresholdController:
    """Mutable P-controller over a probability threshold τ with pause hysteresis.

    Parameters
    ----------
    tau:
        Current threshold. Caller seeds with the warmup-derived value.
    sigma_max:
        Confidence gate — entries blocked when sigma_ve > sigma_max.
    r_star_init:
        Target rate; used to warm-start both EWMAs so the controller starts
        near zero error rather than jumping on the first observation.
    tau_floor, tau_ceil:
        Hard clips on τ. ``tau_floor`` is the EV-breakeven floor below which
        a positive-edge gate cannot exist; ``tau_ceil`` keeps τ from running
        away.
    gamma:
        P-controller gain on (r_hat_ewma - r_star_ewma).
    lambda_decay:
        EWMA forget factor. 2e-3 ≈ 350-bar half-life ≈ 4 days at 20m bars.
    dtau_max:
        Per-update clip on |Δτ|.
    r_min, r_resume:
        Pause hysteresis. Pause when ``r_star_ewma < r_min``; resume when
        ``r_star_ewma > r_resume``. ``r_resume`` MUST be > ``r_min`` for
        true hysteresis.
    """

    tau: float
    sigma_max: float
    r_star_init: float
    tau_floor: float = 0.524
    tau_ceil: float = 0.95
    gamma: float = 0.005
    lambda_decay: float = 2e-3
    dtau_max: float = 0.001
    r_min: float = 0.02
    r_resume: float = 0.05

    # Mutable state — initialized in __post_init__.
    r_hat_ewma: float = field(init=False)
    r_star_ewma: float = field(init=False)
    paused: bool = field(init=False, default=False)
    last_sigma_ve: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Warm-start both EWMAs at the target — error starts at zero so the
        # P-controller doesn't jerk τ on the first observation.
        self.r_hat_ewma = float(self.r_star_init)
        self.r_star_ewma = float(self.r_star_init)
        if self.r_resume <= self.r_min:
            raise ValueError(
                f"r_resume ({self.r_resume}) must be > r_min ({self.r_min}) "
                "for hysteresis to be well-defined"
            )
        if not (0.0 <= self.tau_floor <= self.tau_ceil <= 1.0):
            raise ValueError(
                f"require 0 <= tau_floor ({self.tau_floor}) <= tau_ceil "
                f"({self.tau_ceil}) <= 1"
            )
        if self.dtau_max < 0:
            raise ValueError(f"dtau_max must be >= 0; got {self.dtau_max}")

    # ---- Decisions ---------------------------------------------------------

    def should_enter(
        self,
        p_online: float,
        sigma_ve: Optional[float],
    ) -> bool:
        """True if the controller would open at the current state.

        Side-effect: caches sigma_ve for later inspection via ``get_state``.
        """
        self.last_sigma_ve = float(sigma_ve) if sigma_ve is not None else None
        if self.paused:
            return False
        if sigma_ve is not None and float(sigma_ve) > self.sigma_max:
            return False
        return float(p_online) >= self.tau

    # ---- Updates -----------------------------------------------------------

    def observe(self, entered: bool, label_matured: Optional[int]) -> None:
        """Advance the EWMAs and τ by one step.

        Should be called exactly once per bar with the PREVIOUS bar's
        ``entered`` flag and any newly-matured label (None if no label
        matured this bar).
        """
        # r_hat tracks the entered indicator on every bar.
        e = 1.0 if entered else 0.0
        self.r_hat_ewma = (
            (1.0 - self.lambda_decay) * self.r_hat_ewma
            + self.lambda_decay * e
        )

        # r_star + pause hysteresis update only when a label arrives.
        if label_matured is not None:
            y = float(int(label_matured))
            self.r_star_ewma = (
                (1.0 - self.lambda_decay) * self.r_star_ewma
                + self.lambda_decay * y
            )
            if not self.paused and self.r_star_ewma < self.r_min:
                self.paused = True
            elif self.paused and self.r_star_ewma > self.r_resume:
                self.paused = False

        # τ update is suppressed while paused — no point chasing an error
        # signal we won't act on.
        if not self.paused:
            err = self.r_hat_ewma - self.r_star_ewma
            dtau = self.gamma * err
            if dtau > self.dtau_max:
                dtau = self.dtau_max
            elif dtau < -self.dtau_max:
                dtau = -self.dtau_max
            self.tau = max(self.tau_floor, min(self.tau_ceil, self.tau + dtau))

    # ---- Inspection --------------------------------------------------------

    def get_state(self) -> dict:
        """Snapshot for the engine's per-bar state collector."""
        return {
            "tau": float(self.tau),
            "r_hat": float(self.r_hat_ewma),
            "r_star": float(self.r_star_ewma),
            "paused": bool(self.paused),
            "sigma_ve": self.last_sigma_ve,
        }


__all__ = ["AdaptiveThresholdController"]
