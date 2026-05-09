"""save/load — Pipeline + Broker state with manifest.

Format: tar.zst of {manifest.json, pipeline.pkl, broker.pkl}.
Manifest carries dep versions; load() rejects mismatches unless force=True.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import io
import json
import logging
import pickle
import sys
import tarfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from wagie import __version__ as WAGIE_VERSION


logger = logging.getLogger(__name__)


PINNED_DEPS = ("river", "polars", "numba", "pydantic", "catboost",
               "skfolio", "skrub", "crepes", "mlfinpy", "numpy")


@dataclass
class Manifest:
    wagie_version: str
    python_version: str
    code_sha: str
    deps: dict
    config_hash: Optional[str]
    seed: Optional[int]
    n_seen: int
    state_hash: str

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), indent=2, sort_keys=True).encode()

    @classmethod
    def from_json(cls, b: bytes) -> "Manifest":
        d = json.loads(b)
        return cls(**d)


def _get_dep_versions() -> dict:
    versions = {}
    for d in PINNED_DEPS:
        try:
            versions[d] = md.version(d)
        except md.PackageNotFoundError:
            versions[d] = "missing"
    return versions


def _get_code_sha() -> str:
    """Best-effort git SHA of the wagie module."""
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=Path(__file__).parent)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def save(
    pipeline,
    broker,
    path: str | Path,
    *,
    config_hash: Optional[str] = None,
    seed: Optional[int] = None,
    n_seen: int = 0,
) -> Path:
    """Write a tar of {manifest.json, pipeline.pkl, broker.pkl}."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pipe_state = pipeline.state_dict() if hasattr(pipeline, "state_dict") else {}
    broker_state = (
        broker.finalize().__dict__ if hasattr(broker, "finalize")
        else {"empty": True}
    )

    state_hash = pipeline.state_hash().hex() if hasattr(pipeline, "state_hash") else "unknown"

    manifest = Manifest(
        wagie_version=WAGIE_VERSION,
        python_version=sys.version.split()[0],
        code_sha=_get_code_sha(),
        deps=_get_dep_versions(),
        config_hash=config_hash,
        seed=seed,
        n_seen=int(n_seen),
        state_hash=state_hash,
    )

    pipe_pkl = pickle.dumps(pipe_state, protocol=pickle.HIGHEST_PROTOCOL)
    broker_pkl = pickle.dumps(broker_state, protocol=pickle.HIGHEST_PROTOCOL)
    man_bytes = manifest.to_json()

    with tarfile.open(out, "w:gz") as tf:
        for name, data in [
            ("manifest.json", man_bytes),
            ("pipeline.pkl", pipe_pkl),
            ("broker.pkl", broker_pkl),
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    logger.info(f"wagie.save: wrote {out} (state_hash={state_hash[:16]})")
    return out


def load(path: str | Path, *, force: bool = False) -> tuple[dict, dict, Manifest]:
    """Read manifest + pipeline state + broker state.

    Returns (pipe_state, broker_state, manifest). The caller is responsible for
    constructing fresh Pipeline/Broker objects and calling .load_state_dict().
    Raises if dep versions don't match unless force=True.
    """
    p = Path(path)
    with tarfile.open(p, "r:gz") as tf:
        man_b = tf.extractfile("manifest.json").read()
        pipe_b = tf.extractfile("pipeline.pkl").read()
        broker_b = tf.extractfile("broker.pkl").read()
    manifest = Manifest.from_json(man_b)
    pipe_state = pickle.loads(pipe_b)
    broker_state = pickle.loads(broker_b)

    cur_deps = _get_dep_versions()
    mismatches = []
    for k, v in manifest.deps.items():
        cur = cur_deps.get(k, "missing")
        if cur != v:
            mismatches.append(f"{k}: manifest={v} cur={cur}")
    if mismatches and not force:
        raise RuntimeError(
            "wagie.load: dep version mismatch:\n  " + "\n  ".join(mismatches)
            + "\n  pass force=True to ignore."
        )
    if mismatches:
        logger.warning(f"wagie.load: dep mismatches (force=True):\n  " + "\n  ".join(mismatches))
    return pipe_state, broker_state, manifest


__all__ = ["save", "load", "Manifest"]
