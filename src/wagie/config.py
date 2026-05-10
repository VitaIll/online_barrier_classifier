"""WagieConfig — pydantic-validated nested config for the engine.

The single source of truth for engine configuration:
    WagieConfig(data, model, strategy, broker, runtime, cv)

Each subconfig is a pydantic BaseModel with `extra="forbid"` — typos are errors.
End users go through `wagie experiment run <spec.yaml>`; this config is wrapped
by `ExperimentSpec.wagie`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BrokerCostModel(BaseModel):
    """Composable trading-cost model.

    Total per-side cost in log-units is approximately:
        (taker_fee_bps + 0.5 * spread_bps + slippage_bps_per_unit_size * size) * 1e-4

    Defaults match the legacy flat-1bp/side behaviour so existing YAML keeps
    working when only `cost_bps: 1.0` is specified at the broker level.
    """

    taker_fee_bps: float = 1.0
    spread_bps: float = 0.0
    slippage_bps_per_unit_size: float = 0.0

    model_config = ConfigDict(extra="forbid")

    def cost_log_per_side(self, size: float = 1.0) -> float:
        """Per-side cost in log-units. `size` ∈ [0,1] (Probability)."""
        bps = (
            float(self.taker_fee_bps)
            + 0.5 * float(self.spread_bps)
            + float(self.slippage_bps_per_unit_size) * float(size)
        )
        return bps * 1e-4


class DataConfig(BaseModel):
    parquet_path: str
    m_minutes: int = 20
    start_ts_ms: Optional[int] = None
    end_ts_ms: Optional[int] = None
    symbol: str = "BTCUSDT"
    venue: str = "binance"

    model_config = ConfigDict(extra="forbid")


class ARFSubConfig(BaseModel):
    n_models: int = 10
    max_features: str = "sqrt"
    lambda_value: float = 6.0
    seed: int = 42

    model_config = ConfigDict(extra="forbid")


class ACISubConfig(BaseModel):
    alphas: tuple[float, ...] = (0.05, 0.10, 0.20)
    gamma: float = 0.01
    n_regimes: int = 3
    q_init: float = 0.5
    regime_feature: str = "regime_id"
    q_init_by_regime: Optional[dict] = None

    model_config = ConfigDict(extra="forbid")


class ModelConfig(BaseModel):
    catboost_path: Optional[str] = None
    catboost_ensemble_n: int = 1
    selected_features_path: Optional[str] = None
    arf: ARFSubConfig = Field(default_factory=ARFSubConfig)
    aci: ACISubConfig = Field(default_factory=ACISubConfig)

    model_config = ConfigDict(extra="forbid")


class StrategyConfig(BaseModel):
    kind: Literal["pure_conformal", "ev_calibrated_size"] = "pure_conformal"
    alpha: float = 0.10
    k: Optional[float] = None
    tau: Optional[float] = None
    layer: Optional[str] = None
    extra: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class BrokerConfig(BaseModel):
    take_profit_log: float = 0.0041113
    # Optional: None ⇒ no stop-loss branch (c_stop = ∞ in audit terminology).
    stop_loss_log: Optional[float] = 0.0041113   # symmetric (D1); None => no SL
    cost_bps: float = 1.0                        # legacy flat per-side bps
    cost_model: Optional[BrokerCostModel] = None # richer cost model; if set, overrides cost_bps
    tie_break: Literal[
        "pessimistic_sl_first",
        "optimistic_tp_first",
        "probabilistic_hl_bridge",
    ] = "pessimistic_sl_first"
    execution_latency_minutes: int = 1  # D3
    expiry_minutes: int = 20
    inventory_cap: int = 5             # P1: default 5
    label_alpha: float = 0.0041113     # one-sided excursion (D2)

    model_config = ConfigDict(extra="forbid")

    def resolved_cost_model(self) -> BrokerCostModel:
        """Return a BrokerCostModel — the explicit one if provided, else
        a flat cost from cost_bps."""
        if self.cost_model is not None:
            return self.cost_model
        return BrokerCostModel(taker_fee_bps=float(self.cost_bps))


class CVConfig(BaseModel):
    n_folds: int = 10
    n_test_folds: int = 2
    purged_size: int = 1
    embargo_size: int = 5              # D4: 5 decision periods

    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(BaseModel):
    seed: int = 42
    warmup_samples: int = 96
    capture_audit: bool = False
    output_dir: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class WagieConfig(BaseModel):
    """Nested pydantic config — the single source of truth for engine config."""

    data: DataConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WagieConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if "aci_alphas" in data.get("model", {}).get("aci", {}):
            data["model"]["aci"]["alphas"] = tuple(data["model"]["aci"]["alphas"])
        return cls.model_validate(data)

    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.model_dump_json().encode())
        return h.hexdigest()[:16]


__all__ = [
    "WagieConfig", "DataConfig", "ModelConfig", "StrategyConfig",
    "BrokerConfig", "BrokerCostModel", "CVConfig", "RuntimeConfig",
    "ARFSubConfig", "ACISubConfig",
]
