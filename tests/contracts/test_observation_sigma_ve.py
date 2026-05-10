"""Contract tests for the additive ``Observation.sigma_ve`` field.

The new strategy gates on virtual-ensemble dispersion. The schema must:
  - default ``sigma_ve`` to None when not set
  - round-trip set + replace cleanly via the immutable ``with_sigma_ve``
  - leave existing ``with_*`` updates untouched (commutativity)
  - never break the existing ``to_dict`` flat view
"""

from __future__ import annotations

import dataclasses

import pytest

from wagie.core.event import DecisionBar
from wagie.core.identity import DEFAULT_INSTRUMENT
from wagie.core.numeric import Price, Probability, Quantity
from wagie.core.observation import Observation
from wagie.core.time import Duration, Timestamp


_NS_PER_MIN = 60_000_000_000


def _bar(close: float = 100.0) -> DecisionBar:
    return DecisionBar(
        ts_init=Timestamp(_NS_PER_MIN),
        instrument=DEFAULT_INSTRUMENT,
        open=Price(close), high=Price(close * 1.01), low=Price(close * 0.99),
        close=Price(close), volume=Quantity(1.0),
        duration=Duration.from_minutes(20), segment_id=0,
    )


def test_observation_default_sigma_ve_is_none() -> None:
    obs = Observation(bar=_bar())
    assert obs.sigma_ve is None


def test_with_sigma_ve_returns_new_immutable_observation() -> None:
    obs = Observation(bar=_bar())
    obs2 = obs.with_sigma_ve(0.123)
    # Frozen dataclass: original untouched.
    assert obs.sigma_ve is None
    assert obs2.sigma_ve == pytest.approx(0.123)
    # Identity differs, type matches.
    assert obs is not obs2
    assert isinstance(obs2, Observation)


def test_with_sigma_ve_coerces_to_float() -> None:
    obs = Observation(bar=_bar()).with_sigma_ve(1)
    assert isinstance(obs.sigma_ve, float)
    assert obs.sigma_ve == 1.0


def test_with_sigma_ve_round_trip_replace() -> None:
    """Setting then re-setting yields the latest value."""
    obs = Observation(bar=_bar()).with_sigma_ve(0.1).with_sigma_ve(0.5)
    assert obs.sigma_ve == pytest.approx(0.5)


def test_sigma_ve_independent_of_other_with_methods() -> None:
    """Ensure with_sigma_ve commutes with with_p_online / with_regime."""
    o1 = (Observation(bar=_bar())
          .with_sigma_ve(0.7)
          .with_p_online(Probability(0.4))
          .with_regime(2))
    o2 = (Observation(bar=_bar())
          .with_regime(2)
          .with_p_online(Probability(0.4))
          .with_sigma_ve(0.7))
    assert o1.sigma_ve == o2.sigma_ve == pytest.approx(0.7)
    assert float(o1.p_online) == float(o2.p_online) == pytest.approx(0.4)
    assert o1.regime_id == o2.regime_id == 2


def test_to_dict_unchanged_when_sigma_ve_none() -> None:
    """Pre-existing ``to_dict`` view must not gain new keys when sigma_ve is unset."""
    d = Observation(bar=_bar()).to_dict()
    assert "sigma_ve" not in d


def test_dataclass_replace_preserves_sigma_ve() -> None:
    """Generic ``dataclasses.replace`` keeps sigma_ve through unrelated edits."""
    obs = Observation(bar=_bar()).with_sigma_ve(0.42)
    obs2 = dataclasses.replace(obs, regime_id=7)
    assert obs2.sigma_ve == pytest.approx(0.42)
    assert obs2.regime_id == 7
