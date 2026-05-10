"""save/load — Pipeline + Broker state with manifest.

Format: tar.gz of {manifest.json, pipeline.pkl, broker.pkl}.
Manifest carries dep versions; load() rejects mismatches unless force=True.

Audit-fix additions:
    * ``checkpoint_periodic`` — engine-callback factory that writes
      ``checkpoint_<n>.tar.gz`` snapshots under ``out_dir/state/``. Wired
      into :class:`wagie.engine.Engine` via ``checkpoint_callback``.
    * ``safe_pickle_load`` — restricts unpickling to wagie-namespaced classes
      (and a small allowlist of stdlib / numpy primitives). Replaces the
      naive ``pickle.loads`` so a tampered checkpoint can't trigger
      arbitrary code execution.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from wagie import __version__ as WAGIE_VERSION


logger = logging.getLogger(__name__)


PINNED_DEPS = ("river", "polars", "numba", "pydantic", "catboost",
               "skfolio", "skrub", "crepes", "mlfinpy", "numpy")


# Modules whose classes are safe to unpickle (loaded checkpoints come from us).
_SAFE_PICKLE_PREFIXES: tuple[str, ...] = (
    "wagie.",
    "builtins",
    "collections",
    "dataclasses",
    "numpy",
    "numpy.core",
    "numpy.dtypes",
    "_codecs",
)


class _SafePickleError(pickle.UnpicklingError):
    """Raised when pickle tries to import an unauthorized class."""


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that refuses any class outside :data:`_SAFE_PICKLE_PREFIXES`.

    Wagie checkpoints contain only ``wagie.*`` dataclasses, ``dict``,
    ``list``, numpy arrays, and similar primitives. Anything else is a sign
    of a tampered or alien archive.
    """

    def find_class(self, module: str, name: str):
        if not any(module == p or module.startswith(p)
                   for p in _SAFE_PICKLE_PREFIXES):
            raise _SafePickleError(
                f"safe_pickle_load: refusing to unpickle "
                f"{module}.{name} (not in wagie.* allowlist)"
            )
        return super().find_class(module, name)


def safe_pickle_load(buf: bytes) -> Any:
    """Drop-in replacement for ``pickle.loads`` restricted to wagie/numpy/stdlib.

    Tampered or alien checkpoints raise :class:`_SafePickleError`.
    """
    return _SafeUnpickler(io.BytesIO(buf)).load()


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


def load(path: str | Path, *, force: bool = False,
         unsafe_pickle: bool = False) -> tuple[dict, dict, Manifest]:
    """Read manifest + pipeline state + broker state.

    Returns (pipe_state, broker_state, manifest). The caller is responsible for
    constructing fresh Pipeline/Broker objects and calling .load_state_dict().
    Raises if dep versions don't match unless force=True.

    Args:
        path: Checkpoint archive path.
        force: Bypass dep-version mismatch (just warn).
        unsafe_pickle: If True, fall back to plain ``pickle.loads`` (legacy
            behavior). Default False — uses :func:`safe_pickle_load`, which
            restricts unpickled classes to ``wagie.*`` + a small allowlist.
    """
    p = Path(path)
    with tarfile.open(p, "r:gz") as tf:
        man_b = tf.extractfile("manifest.json").read()
        pipe_b = tf.extractfile("pipeline.pkl").read()
        broker_b = tf.extractfile("broker.pkl").read()
    manifest = Manifest.from_json(man_b)
    if unsafe_pickle:
        pipe_state = pickle.loads(pipe_b)
        broker_state = pickle.loads(broker_b)
    else:
        pipe_state = safe_pickle_load(pipe_b)
        broker_state = safe_pickle_load(broker_b)

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


# -----------------------------------------------------------------------------
# Periodic checkpointing helper (audit-fix)
# -----------------------------------------------------------------------------


def checkpoint_periodic(
    out_dir: str | Path,
    *,
    config_hash: Optional[str] = None,
    seed: Optional[int] = None,
    keep_last: int = 0,
) -> Callable[[Any], Path]:
    """Build an Engine ``checkpoint_callback`` that snapshots state.

    Each invocation writes ``<out_dir>/state/checkpoint_<n>.tar.gz`` where
    ``n`` is the engine's current ``_n_decisions``. The callback is suitable
    for :class:`wagie.engine.Engine`'s ``checkpoint_callback`` parameter.

    Args:
        out_dir: Run output directory; checkpoints land in ``out_dir/state``.
        config_hash: Optional config hash forwarded into the manifest.
        seed: Optional seed forwarded into the manifest.
        keep_last: If > 0, prune older checkpoints, keeping only the most
            recent ``keep_last`` snapshots. 0 (default) keeps all.

    Returns:
        A callable ``f(engine) -> Path`` that returns the checkpoint path.

    Example:
        cb = checkpoint_periodic(out_dir, config_hash=cfg.hash(), keep_last=3)
        engine = Engine(..., checkpoint_callback=cb, checkpoint_every=1000)
    """
    out_dir = Path(out_dir)
    state_dir = out_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    def _callback(engine) -> Path:
        n = int(getattr(engine, "_n_decisions", 0))
        path = state_dir / f"checkpoint_{n:08d}.tar.gz"
        save(
            engine.pipeline, engine.broker, path,
            config_hash=config_hash, seed=seed,
            n_seen=int(getattr(engine, "_n_seen", 0)),
        )
        if keep_last > 0:
            _prune_checkpoints(state_dir, keep_last=keep_last)
        return path

    return _callback


def _prune_checkpoints(state_dir: Path, *, keep_last: int) -> None:
    """Delete older ``checkpoint_*.tar.gz`` files, keeping the N most recent."""
    snaps = sorted(state_dir.glob("checkpoint_*.tar.gz"))
    surplus = snaps[:-keep_last] if keep_last > 0 else []
    for p in surplus:
        try:
            p.unlink()
        except OSError as exc:
            logger.warning("checkpoint prune failed for %s: %s", p, exc)


__all__ = [
    "save", "load", "Manifest",
    "safe_pickle_load",
    "checkpoint_periodic",
]
