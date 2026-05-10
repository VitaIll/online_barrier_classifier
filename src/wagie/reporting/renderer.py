"""ReportRenderer — the SINGLE canonical report assembly point.

Default report root: ``artifacts/report/``. Layout::

    artifacts/report/
    ├── index.html              # rendered each run
    ├── manifest.json           # what each section contributed (file list,
    │                           #   html fragment, metadata, order)
    ├── figs/<section>/*.{png,svg,html}
    ├── tables/<section>.json   # raw tables a section may want to expose
    ├── spec.yaml               # snapshot of the spec
    ├── metrics.json            # MetricsBattery dump
    ├── state/
    │   ├── pipeline_state_hash.txt
    │   └── rebuild_bundle.pkl  # minimal inputs needed for per-section rebuild
    └── _archive/
        └── <ts>_<hash>.zip     # last 10 archived snapshots

A render call:
    1. Optionally archives the current contents to ``_archive/`` (zip) and
       prunes oldest so only the last ``archive_keep`` remain.
    2. Wipes the live tree (preserving ``_archive/``).
    3. Runs each registered ``SectionEmitter`` in ascending ``order``.
    4. Writes ``manifest.json`` + ``index.html`` from a single Jinja template.
    5. Persists the rebuild bundle so ``rebuild(sections=...)`` can rerun a
       specific section without touching the engine.

The render is idempotent w.r.t. the inputs — a re-render with the same
inputs produces a manifest with the same files (orphans pruned). Side
experiments use a different ``report_root`` and never touch the canonical
location.
"""

from __future__ import annotations

import logging
import pickle
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .archive import (
    ARCHIVE_DIRNAME,
    DEFAULT_KEEP,
    archive_current,
    wipe_payload,
)
from .manifest import ReportManifest, RunMeta, SectionRecord
from .sections import (
    EmitterContext,
    SectionEmitter,
    _autoload_builtin_sections,
    all_emitters,
    get_emitter,
)


logger = logging.getLogger(__name__)


DEFAULT_REPORT_ROOT = Path("artifacts/report")
TEMPLATE_NAME = "index.html.j2"
REBUILD_BUNDLE_NAME = "rebuild_bundle.pkl"


@dataclass
class RebuildBundle:
    """Minimal pickle-safe payload kept under ``state/`` so per-section
    rebuilds work without re-running the engine.

    Only fields a section emitter might want are kept. The full
    ``EngineResult`` is intentionally NOT pickled — it carries Event
    objects and other non-portable state.
    """

    metrics: dict
    spec_dict: dict
    run_meta: dict
    use_plotly: bool = True
    # Streaming traces from the engine that emitters use to produce charts
    p_online_history: list[float] = field(default_factory=list)
    label_history: list[int] = field(default_factory=list)
    regime_history: list[int] = field(default_factory=list)
    # Trading PnL series (per-fill log return) — used for equity/drawdown/dist
    pnl_log: list[float] = field(default_factory=list)
    # CV per-fold rows when in CV mode
    cv_per_fold: Optional[list[dict]] = None
    cv_pbo: Optional[float] = None
    # Adaptive-threshold per-bar strategy state (additive). Empty for legacy
    # strategies that don't implement get_state().
    tau_history: list[float] = field(default_factory=list)
    r_hat_ewma_history: list[float] = field(default_factory=list)
    r_star_ewma_history: list[float] = field(default_factory=list)
    paused_history: list[bool] = field(default_factory=list)
    sigma_ve_history: list[float] = field(default_factory=list)
    inventory_size_history: list[int] = field(default_factory=list)
    hold_age_max_history: list[int] = field(default_factory=list)
    warmup_calibration: Optional[dict] = None


# =============================================================================
# Helpers
# =============================================================================

def _resolve_template_dir() -> Path:
    """Return the package-bundled template directory."""
    return Path(__file__).resolve().parent / "templates"


def _safe_iter_files(root: Path, *, exclude_dirs: Iterable[str]) -> list[Path]:
    """Yield every file under ``root`` skipping the given top-level directory
    names."""
    excluded = set(exclude_dirs)
    out: list[Path] = []
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if len(rel.parts) > 0 and rel.parts[0] in excluded:
            continue
        out.append(p)
    return out


def _git_short_sha() -> str:
    """Best-effort git short SHA (no error if not in a repo or git absent)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        )
        return out.decode().strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _build_bundle(
    *, metrics: dict, spec_dict: dict, run_meta: RunMeta,
    use_plotly: bool, engine_result: Any = None,
    cv_per_fold: Optional[list[dict]] = None,
    cv_pbo: Optional[float] = None,
) -> RebuildBundle:
    """Snapshot just the fields needed to rebuild any section."""
    p_hist: list[float] = []
    y_hist: list[int] = []
    r_hist: list[int] = []
    pnl: list[float] = []
    tau_h: list[float] = []
    rhat_h: list[float] = []
    rstar_h: list[float] = []
    paused_h: list[bool] = []
    sigma_h: list[float] = []
    inv_h: list[int] = []
    age_h: list[int] = []
    warmup_cal: Optional[dict] = None
    if engine_result is not None:
        p_hist = list(getattr(engine_result, "p_online_history", []) or [])
        y_hist = list(getattr(engine_result, "label_history", []) or [])
        r_hist = list(getattr(engine_result, "regime_history", []) or [])
        ledger = getattr(engine_result, "ledger", None)
        if ledger is not None:
            pnl = [float(getattr(f, "pnl_log_net", 0.0)) for f in (ledger.fills or [])]
        tau_h = list(getattr(engine_result, "tau_history", []) or [])
        rhat_h = list(getattr(engine_result, "r_hat_ewma_history", []) or [])
        rstar_h = list(getattr(engine_result, "r_star_ewma_history", []) or [])
        paused_h = list(getattr(engine_result, "paused_history", []) or [])
        sigma_h = list(getattr(engine_result, "sigma_ve_history", []) or [])
        inv_h = list(getattr(engine_result, "inventory_size_history", []) or [])
        age_h = list(getattr(engine_result, "hold_age_max_history", []) or [])
        wc = getattr(engine_result, "warmup_calibration", None)
        if wc is not None:
            warmup_cal = dict(wc) if isinstance(wc, dict) else wc
    return RebuildBundle(
        metrics=dict(metrics or {}),
        spec_dict=dict(spec_dict or {}),
        run_meta=run_meta.to_dict(),
        use_plotly=bool(use_plotly),
        p_online_history=p_hist,
        label_history=y_hist,
        regime_history=r_hist,
        pnl_log=pnl,
        cv_per_fold=cv_per_fold,
        cv_pbo=cv_pbo,
        tau_history=tau_h,
        r_hat_ewma_history=rhat_h,
        r_star_ewma_history=rstar_h,
        paused_history=paused_h,
        sigma_ve_history=sigma_h,
        inventory_size_history=inv_h,
        hold_age_max_history=age_h,
        warmup_calibration=warmup_cal,
    )


# =============================================================================
# Renderer
# =============================================================================

@dataclass
class ReportRenderer:
    """The ONE renderer. Dispatches to registered ``SectionEmitter``s.

    Use ``render(...)`` for a full rebuild from a fresh engine result.
    Use ``rebuild(sections=...)`` to redo only specific sections from
    the persisted rebuild bundle.
    """

    report_root: Path = DEFAULT_REPORT_ROOT
    archive_keep: int = DEFAULT_KEEP
    enable_archive: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        *,
        engine_result: Any,
        metrics: dict,
        spec_dict: dict,
        run_meta: RunMeta,
        cv_result: Any = None,
        use_plotly: bool = True,
        title: Optional[str] = None,
        archive: Optional[bool] = None,
    ) -> Path:
        """Full render. Wipes and rewrites ``self.report_root``.

        Returns the path to ``index.html``.
        """
        _autoload_builtin_sections()
        report_root = Path(self.report_root)
        report_root.mkdir(parents=True, exist_ok=True)

        # 1. Archive current state (if enabled).
        do_archive = self.enable_archive if archive is None else archive
        if do_archive:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            archive_current(
                report_root, timestamp=ts, spec_hash=run_meta.spec_hash,
                max_keep=self.archive_keep,
            )

        # 2. Wipe live payload.
        wipe_payload(report_root, preserve_archive=True)

        # 3. Set up dirs.
        figs_dir = report_root / "figs"
        tables_dir = report_root / "tables"
        state_dir = report_root / "state"
        for d in (figs_dir, tables_dir, state_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 4. Stamp git sha onto run_meta if absent.
        if not run_meta.git_sha:
            run_meta.git_sha = _git_short_sha()
        run_meta.use_plotly = bool(use_plotly)

        # 5. Run all emitters.
        ctx = EmitterContext(
            report_root=report_root, figs_dir=figs_dir, tables_dir=tables_dir,
            metrics=dict(metrics or {}), spec_dict=dict(spec_dict or {}),
            run_meta=run_meta, engine_result=engine_result, cv_result=cv_result,
            use_plotly=bool(use_plotly),
        )
        manifest = ReportManifest(
            generated_at_utc=ReportManifest.now_utc(), run_meta=run_meta,
        )
        for emitter_cls in all_emitters():
            emitter = emitter_cls()
            try:
                record = emitter.emit(ctx)
            except Exception as e:  # one bad emitter must not kill the report
                logger.exception("section emitter %s failed: %s", emitter_cls.name, e)
                record = SectionRecord(
                    name=emitter_cls.name, title=emitter_cls.title,
                    order=emitter_cls.order,
                    html_fragment=(
                        f"<div class=\"empty\"><p><em>section emitter failed: "
                        f"{emitter_cls.name} — see logs.</em></p></div>"
                    ),
                )
            manifest.upsert(record)

        # 6. Persist auxiliary artifacts.
        self._persist_aux(
            report_root, metrics=metrics, spec_dict=spec_dict,
            run_meta=run_meta, engine_result=engine_result,
            use_plotly=use_plotly,
            cv_per_fold=getattr(cv_result, "per_fold", None) if cv_result else None,
            cv_pbo=getattr(cv_result, "pbo", None) if cv_result else None,
        )

        # 7. Write manifest.json + index.html.
        manifest.write(report_root / "manifest.json")
        index_path = self._render_index(
            report_root, manifest, title=title, use_plotly=use_plotly,
        )

        # 8. Prune orphans (defensive — wipe should have handled this).
        self._prune_orphans(report_root, manifest)

        logger.info("report: rendered index=%s", index_path)
        return index_path

    def rebuild(
        self,
        *,
        sections: Optional[Iterable[str]] = None,
        title: Optional[str] = None,
    ) -> Path:
        """Rebuild a subset of sections from the persisted rebuild bundle.

        ``sections`` is a list of section ``name``s. Pass None to rebuild
        all sections. Other sections retain their existing manifest entries.
        Files belonging to rebuilt sections are wiped first; orphans from
        renamed/removed sections are also cleared on the next full render.
        """
        _autoload_builtin_sections()
        report_root = Path(self.report_root)
        manifest_path = report_root / "manifest.json"
        bundle_path = report_root / "state" / REBUILD_BUNDLE_NAME

        prior = ReportManifest.read(manifest_path)
        if prior is None:
            raise FileNotFoundError(
                f"no manifest found at {manifest_path} — run a full render first"
            )
        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"no rebuild bundle at {bundle_path} — engine result was not "
                f"persisted (full render required)"
            )

        with bundle_path.open("rb") as f:
            bundle: RebuildBundle = pickle.load(f)

        run_meta = RunMeta.from_dict(bundle.run_meta)
        run_meta.note = (run_meta.note or "") + " [rebuild]"

        targets = self._resolve_targets(sections, prior)
        figs_dir = report_root / "figs"
        tables_dir = report_root / "tables"

        # Wipe figs/tables for the targeted sections only.
        for name in targets:
            for d in (figs_dir / name, tables_dir / name):
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)

        # Build a fake EngineResult-like object the emitters can consume.
        replay_engine = _ReplayEngineResult.from_bundle(bundle)
        cv_result_stub = (
            _ReplayCVResult(per_fold=bundle.cv_per_fold or [], pbo=bundle.cv_pbo or 0.0)
            if bundle.cv_per_fold else None
        )

        ctx = EmitterContext(
            report_root=report_root, figs_dir=figs_dir, tables_dir=tables_dir,
            metrics=dict(bundle.metrics or {}), spec_dict=dict(bundle.spec_dict or {}),
            run_meta=run_meta, engine_result=replay_engine, cv_result=cv_result_stub,
            use_plotly=bool(bundle.use_plotly),
        )

        # Re-run the targeted emitters.
        for name in targets:
            cls = get_emitter(name)
            if cls is None:
                logger.warning("rebuild: no emitter named %r — leaving stale", name)
                continue
            try:
                record = cls().emit(ctx)
            except Exception as e:
                logger.exception("rebuild: emitter %s failed: %s", name, e)
                record = SectionRecord(
                    name=name, title=cls.title, order=cls.order,
                    html_fragment=(
                        f"<div class=\"empty\"><p><em>rebuild failed: {name}"
                        f" — see logs.</em></p></div>"
                    ),
                )
            prior.upsert(record)

        prior.generated_at_utc = ReportManifest.now_utc()
        prior.write(manifest_path)
        index_path = self._render_index(
            report_root, prior, title=title, use_plotly=bool(bundle.use_plotly),
        )
        logger.info("report: rebuilt sections=%s index=%s", sorted(targets), index_path)
        return index_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_targets(
        self, sections: Optional[Iterable[str]], manifest: ReportManifest,
    ) -> set[str]:
        if sections is None:
            return {cls.name for cls in all_emitters()}
        names = {str(s).strip() for s in sections if str(s).strip()}
        # Validate each name resolves to an emitter (or at least an existing
        # section in the manifest, so caller gets a clear error early).
        unknown = {n for n in names if get_emitter(n) is None
                   and n not in manifest.sections}
        if unknown:
            raise KeyError(
                f"unknown section(s): {sorted(unknown)} — known: "
                f"{sorted(get_emitter(c.name).name for c in all_emitters() if get_emitter(c.name))}"
            )
        return names

    def _persist_aux(
        self, report_root: Path, *, metrics: dict, spec_dict: dict,
        run_meta: RunMeta, engine_result: Any, use_plotly: bool,
        cv_per_fold: Optional[list[dict]] = None,
        cv_pbo: Optional[float] = None,
    ) -> None:
        """Write spec.yaml, metrics.json, state files, and the rebuild bundle."""
        import json as _json
        try:
            import yaml as _yaml
            (report_root / "spec.yaml").write_text(
                _yaml.safe_dump(dict(spec_dict), sort_keys=False,
                                default_flow_style=False),
                encoding="utf-8",
            )
        except Exception:
            (report_root / "spec.yaml").write_text(
                _json.dumps(dict(spec_dict), indent=2, default=str),
                encoding="utf-8",
            )
        (report_root / "metrics.json").write_text(
            _json.dumps(dict(metrics), indent=2, default=str), encoding="utf-8",
        )
        state_dir = report_root / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        if engine_result is not None and hasattr(engine_result, "pipeline_state_hash"):
            try:
                h = engine_result.pipeline_state_hash
                (state_dir / "pipeline_state_hash.txt").write_text(
                    h.hex() if isinstance(h, (bytes, bytearray)) else str(h),
                    encoding="utf-8",
                )
            except Exception:
                pass
        bundle = _build_bundle(
            metrics=metrics, spec_dict=spec_dict, run_meta=run_meta,
            use_plotly=use_plotly, engine_result=engine_result,
            cv_per_fold=cv_per_fold, cv_pbo=cv_pbo,
        )
        try:
            with (state_dir / REBUILD_BUNDLE_NAME).open("wb") as f:
                pickle.dump(bundle, f, protocol=4)
        except (pickle.PicklingError, TypeError) as e:  # pragma: no cover - defensive
            logger.warning("rebuild bundle pickle failed: %s", e)

    def _render_index(
        self, report_root: Path, manifest: ReportManifest, *,
        title: Optional[str] = None, use_plotly: bool = True,
    ) -> Path:
        """Render ``index.html`` from the Jinja template."""
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        template_dir = _resolve_template_dir()
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(disabled_extensions=()),
            trim_blocks=True, lstrip_blocks=True,
        )
        tmpl = env.get_template(TEMPLATE_NAME)

        any_plotly = any(
            r.metadata.get("uses_plotly", False) for r in manifest.sections.values()
        )

        title_str = title or f"wagie report — {manifest.run_meta.spec_name}"
        body = tmpl.render(
            title=title_str,
            run_meta=manifest.run_meta,
            generated_at_utc=manifest.generated_at_utc,
            schema_version=manifest.schema_version,
            sections=manifest.ordered(),
            include_plotly=bool(use_plotly and any_plotly),
        )
        out = report_root / "index.html"
        out.write_text(body, encoding="utf-8")
        return out

    def _prune_orphans(self, report_root: Path, manifest: ReportManifest) -> None:
        """Delete any file under figs/ or tables/ not listed in the manifest."""
        listed = manifest.all_listed_files()
        for sub in ("figs", "tables"):
            root = report_root / sub
            if not root.is_dir():
                continue
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(report_root).as_posix()
                if rel not in listed:
                    try:
                        p.unlink()
                    except OSError:  # pragma: no cover
                        pass
            # Sweep newly-empty subdirs.
            for sub_p in sorted(
                [d for d in root.rglob("*") if d.is_dir()],
                key=lambda d: -len(str(d)),
            ):
                try:
                    sub_p.rmdir()
                except OSError:
                    pass


# =============================================================================
# Replay helpers — minimal stand-ins reconstructed from a RebuildBundle so
# ``rebuild(sections=...)`` can run section emitters without an EngineResult.
# Section emitters that touch attributes outside this surface should fall back
# gracefully (return an empty record).
# =============================================================================

@dataclass
class _ReplayLedger:
    fills: list = field(default_factory=list)


@dataclass
class _ReplayFill:
    pnl_log_net: float = 0.0


@dataclass
class _ReplayEngineResult:
    """A duck-typed stand-in for ``EngineResult`` populated from the bundle."""

    p_online_history: list[float] = field(default_factory=list)
    label_history: list[int] = field(default_factory=list)
    regime_history: list[int] = field(default_factory=list)
    ledger: _ReplayLedger = field(default_factory=_ReplayLedger)
    n_decisions: int = 0
    n_filled: int = 0
    n_actions_approved: int = 0
    n_actions_rejected: int = 0
    pipeline_state_hash: bytes = b""
    # Adaptive-threshold strategy state (additive)
    tau_history: list[float] = field(default_factory=list)
    r_hat_ewma_history: list[float] = field(default_factory=list)
    r_star_ewma_history: list[float] = field(default_factory=list)
    paused_history: list[bool] = field(default_factory=list)
    sigma_ve_history: list[float] = field(default_factory=list)
    inventory_size_history: list[int] = field(default_factory=list)
    hold_age_max_history: list[int] = field(default_factory=list)
    warmup_calibration: Optional[dict] = None

    @classmethod
    def from_bundle(cls, b: "RebuildBundle") -> "_ReplayEngineResult":
        fills = [_ReplayFill(pnl_log_net=float(x)) for x in (b.pnl_log or [])]
        m = b.metrics or {}
        try:
            sh = bytes.fromhex(str(m.get("pipeline_state_hash", "")) or "")
        except ValueError:
            sh = b""
        return cls(
            p_online_history=list(b.p_online_history or []),
            label_history=list(b.label_history or []),
            regime_history=list(b.regime_history or []),
            ledger=_ReplayLedger(fills=fills),
            n_decisions=int(m.get("n_decisions", 0) or 0),
            n_filled=int(m.get("n_filled", 0) or 0),
            n_actions_approved=int(m.get("n_actions_approved", 0) or 0),
            n_actions_rejected=int(m.get("n_actions_rejected", 0) or 0),
            pipeline_state_hash=sh,
            tau_history=list(b.tau_history or []),
            r_hat_ewma_history=list(b.r_hat_ewma_history or []),
            r_star_ewma_history=list(b.r_star_ewma_history or []),
            paused_history=list(b.paused_history or []),
            sigma_ve_history=list(b.sigma_ve_history or []),
            inventory_size_history=list(b.inventory_size_history or []),
            hold_age_max_history=list(b.hold_age_max_history or []),
            warmup_calibration=(
                dict(b.warmup_calibration) if isinstance(b.warmup_calibration, dict)
                else b.warmup_calibration
            ),
        )


@dataclass
class _ReplayCVResult:
    per_fold: list[dict] = field(default_factory=list)
    pbo: float = 0.0
    n_folds: int = 0


__all__ = [
    "DEFAULT_REPORT_ROOT", "REBUILD_BUNDLE_NAME",
    "RebuildBundle", "ReportRenderer",
]
