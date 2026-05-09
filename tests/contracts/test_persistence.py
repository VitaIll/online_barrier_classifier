"""wagie.persistence — save/load round-trip + manifest invariants."""

from __future__ import annotations

import tarfile

import pytest

from wagie.features import BaseBarFeatures, FeatureBuilder, RegimeCuts, RegimeFeature
from wagie.features.catalog import default_streaming_features
from wagie.io.brokers import SimBroker
from wagie.persistence import Manifest, load, save
from wagie.pipeline import (
    LabelBuffer,
    MondrianACICalibrator,
    OnlineARFCorrector,
    Pipeline,
)
from wagie.strategy import PureConformalGate


def _build_pipeline_and_broker():
    base_bar = BaseBarFeatures()
    fb = FeatureBuilder(default_streaming_features()[:3])
    cuts = RegimeCuts(
        feature="parkinson_var_rolling_mean_24",
        edges=(1e-6, 1e-5),
        labels=("low", "med", "high"),
    )
    rg = RegimeFeature(cuts)
    arf = OnlineARFCorrector()
    aci = MondrianACICalibrator()
    lb = LabelBuffer()
    strat = PureConformalGate(name="pcg", alpha=0.10)
    pipeline = Pipeline([base_bar, fb, rg, arf, aci, lb, strat])
    broker = SimBroker(m_minutes=20, inventory_cap=5)
    return pipeline, broker


def test_save_load_round_trip_preserves_state_hash(tmp_path):
    pipeline, broker = _build_pipeline_and_broker()
    snap = tmp_path / "round_trip.tar.gz"
    save(pipeline, broker, snap, config_hash="cfg-deadbeef", seed=42, n_seen=123)

    pipe_state, broker_state, manifest = load(snap)

    expected_hash = pipeline.state_hash().hex()
    assert manifest.state_hash == expected_hash
    assert isinstance(pipe_state, dict)
    assert isinstance(broker_state, dict)
    assert manifest.config_hash == "cfg-deadbeef"
    assert manifest.seed == 42
    assert manifest.n_seen == 123


def test_load_missing_path_raises(tmp_path):
    missing = tmp_path / "does_not_exist.tar.gz"
    with pytest.raises((FileNotFoundError, OSError, tarfile.ReadError)):
        load(missing)


def test_save_with_broker_fills_preserves_count(tmp_path):
    """Broker state captured at save-time reflects the closed-history count."""
    pipeline, broker = _build_pipeline_and_broker()

    # Inject two synthetic 'fills' onto the broker's closed_history. The
    # broker is otherwise inert (no source) — save() reads finalize().__dict__.
    broker._closed_history.extend([{"sentinel": 1}, {"sentinel": 2}])

    snap = tmp_path / "with_fills.tar.gz"
    save(pipeline, broker, snap)
    _, broker_state, _ = load(snap)
    assert "fills" in broker_state
    assert len(broker_state["fills"]) == 2


def test_saved_file_is_real_tar_gz(tmp_path):
    pipeline, broker = _build_pipeline_and_broker()
    snap = tmp_path / "snap.tar.gz"
    save(pipeline, broker, snap)
    assert tarfile.is_tarfile(str(snap))
    # Confirm expected members.
    with tarfile.open(snap, "r:gz") as tf:
        names = set(tf.getnames())
    assert {"manifest.json", "pipeline.pkl", "broker.pkl"}.issubset(names)


def test_manifest_to_from_json_round_trip():
    m = Manifest(
        wagie_version="0.0.0",
        python_version="3.11.0",
        code_sha="abc",
        deps={"river": "0.24.2"},
        config_hash=None,
        seed=None,
        n_seen=0,
        state_hash="deadbeef",
    )
    blob = m.to_json()
    parsed = Manifest.from_json(blob)
    assert parsed == m


def test_load_dep_mismatch_without_force_raises(tmp_path, monkeypatch):
    """If dep versions diverge between save and load, load() must reject."""
    pipeline, broker = _build_pipeline_and_broker()
    snap = tmp_path / "depmismatch.tar.gz"
    save(pipeline, broker, snap)

    # Force the next _get_dep_versions call to return a wrong river version.
    import wagie.persistence as P

    real_versions = P._get_dep_versions()
    real_versions["river"] = "0.0.0-not-installed"
    monkeypatch.setattr(P, "_get_dep_versions", lambda: real_versions)
    with pytest.raises(RuntimeError, match="version mismatch"):
        P.load(snap)
    # force=True should bypass and emit a warning instead of raising.
    pipe_state, broker_state, m = P.load(snap, force=True)
    assert m.state_hash  # manifest still loadable


def test_save_returns_path_and_creates_parent_dir(tmp_path):
    pipeline, broker = _build_pipeline_and_broker()
    nested = tmp_path / "deep" / "nested" / "snap.tar.gz"
    out = save(pipeline, broker, nested)
    assert out == nested
    assert nested.exists()


def test_save_load_handles_no_state_dict_pipeline(tmp_path):
    """Pipeline-like object lacking state_dict still saves (state==empty dict)."""

    class _Bare:
        pass

    class _BareBroker:
        pass

    snap = tmp_path / "bare.tar.gz"
    save(_Bare(), _BareBroker(), snap)
    pipe_state, broker_state, manifest = load(snap, force=True)
    assert pipe_state == {} or isinstance(pipe_state, dict)
    assert manifest.state_hash == "unknown"
