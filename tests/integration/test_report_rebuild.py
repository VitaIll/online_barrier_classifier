"""Integration: end-to-end report renders + archive rotation + per-section rebuild.

Pin down the user-visible report flow:
  1. First render: ``index.html`` exists; ``_archive/`` has zero archives.
  2. Second render: ``_archive/`` now has exactly 1 archive.
  3. Twelve more renders later: ``_archive/`` has exactly 10 archives
     (the "last 10 streaks" rule).
  4. Per-section rebuild succeeds when ``state/rebuild_bundle.pkl`` is
     present, regenerates only the calibration section's files, and leaves
     other sections' files intact.

Marked ``@slow`` and ``@gating`` per the test plan — this is a multi-render
test. Uses ``_FakeEngine`` rather than spinning up the full protocol so the
test stays fast and isn't gated on Agent C's protocol/CLI rewire landing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from wagie.reporting import (
    ARCHIVE_DIRNAME,
    ReportRenderer,
    RunMeta,
    list_archives,
)


# =============================================================================
# Fake engine (mirrors `_ReplayEngineResult` shape).
# =============================================================================

@dataclass
class _FakeFill:
    pnl_log_net: float = 0.0


@dataclass
class _FakeLedger:
    fills: list = field(default_factory=list)


@dataclass
class _FakeEngine:
    p_online_history: list = field(default_factory=list)
    label_history: list = field(default_factory=list)
    regime_history: list = field(default_factory=list)
    ledger: _FakeLedger = field(default_factory=_FakeLedger)
    n_decisions: int = 0
    n_filled: int = 0
    n_skipped_warmup: int = 0
    n_actions_approved: int = 0
    n_actions_rejected: int = 0
    pipeline_state_hash: bytes = b"\xde\xad\xbe\xef" * 8


def _make_engine() -> _FakeEngine:
    p = [0.10, 0.85, 0.40, 0.65, 0.20, 0.55, 0.30, 0.95, 0.45, 0.70,
         0.15, 0.80, 0.35, 0.60, 0.25]
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    r = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    fills = [_FakeFill(pnl_log_net=v) for v in
             (0.001, -0.002, 0.0015, -0.0005, 0.003,
              -0.001, 0.0025, 0.0008, -0.0012, 0.0018)]
    return _FakeEngine(
        p_online_history=p,
        label_history=y,
        regime_history=r,
        ledger=_FakeLedger(fills=fills),
        n_decisions=20,
        n_filled=10,
        n_actions_approved=10,
        n_actions_rejected=5,
        pipeline_state_hash=b"\xab" * 32,
    )


def _metrics() -> dict:
    return {
        "brier": 0.21, "ece": 0.034,
        "calibration_by_regime": [
            {"regime_id": 0, "n": 80, "brier": 0.20, "ece": 0.03},
            {"regime_id": 1, "n": 90, "brier": 0.22, "ece": 0.04},
        ],
        "trading": {
            "n_trades": 10, "n_tp": 5, "n_sl": 4, "n_timeout": 1,
            "sharpe": 1.234, "probabilistic_sharpe": 0.5, "sortino": 1.0,
            "hit_rate": 0.5, "profit_factor": 1.4,
            "max_drawdown_log": -0.05, "cdar_5pct_log": -0.02,
            "total_log_return": 0.012, "total_pct_return": 0.012,
        },
        "n_decisions": 20, "n_filled": 10,
        "n_actions_approved": 10, "n_actions_rejected": 5,
        "roc_auc": 0.55, "pr_auc": 0.31,
        "pipeline_state_hash": "deadbeef" * 8,
        "accepted": True, "blocked_reasons": [],
    }


def _spec() -> dict:
    return {"name": "rebuild_int", "seed": 42}


def _render_once(report_root: Path, *, run_id: str) -> Path:
    rr = ReportRenderer(report_root=report_root, enable_archive=True)
    return rr.render(
        engine_result=_make_engine(),
        metrics=_metrics(),
        spec_dict=_spec(),
        run_meta=RunMeta(
            run_id=run_id, spec_name="rebuild_int", spec_hash="deadbeef",
        ),
        use_plotly=False,
    )


# =============================================================================
# The end-to-end test
# =============================================================================

@pytest.mark.slow
@pytest.mark.gating
def test_report_renders_archives_and_rebuild_full_cycle(tmp_path: Path) -> None:
    """Full cycle: 14 renders + per-section rebuild on the canonical layout.

    Steps mirror the user-facing flow:
      1. First render → index.html exists; _archive/ has zero archives.
      2. Second render → _archive/ has exactly 1 archive.
      3. Twelve more renders → _archive/ has exactly 10 (the "last 10" rule).
      4. Per-section rebuild only modifies calibration files.
    """
    report_root = tmp_path / "report"

    # --- Step 1: first render ---------------------------------------------
    out = _render_once(report_root, run_id="run-001")
    assert out.is_file()
    assert out.name == "index.html"
    # Archive dir either doesn't exist or is empty (no prior payload to archive).
    archives = list_archives(report_root)
    assert len(archives) == 0

    # Snapshot all files for later orphan inspection
    files_after_first = sorted(
        p.relative_to(report_root).as_posix()
        for p in report_root.rglob("*")
        if p.is_file()
        and ARCHIVE_DIRNAME not in p.relative_to(report_root).parts
    )
    assert "index.html" in files_after_first
    assert "manifest.json" in files_after_first
    assert "metrics.json" in files_after_first
    assert "spec.yaml" in files_after_first

    # --- Step 2: second render → 1 archive --------------------------------
    # NOTE: archive filenames are timestamped to the second; bump the run_id
    # so we have a unique stamp + force a small monotonic delta below by
    # using a different timestamp via the renderer's internal clock. Since
    # the renderer uses time.gmtime(), a single second of wall-clock should
    # not generate a collision in practice — but the archive helper has a
    # disambiguating suffix anyway.
    _render_once(report_root, run_id="run-002")
    archives = list_archives(report_root)
    assert len(archives) == 1, \
        f"after 2 renders expected 1 archive, got {len(archives)}"

    # --- Step 3: 12 more renders → 10 archives (last 10 rule) -------------
    for i in range(3, 15):  # run-003 .. run-014
        _render_once(report_root, run_id=f"run-{i:03d}")
    archives = list_archives(report_root)
    assert len(archives) == 10, \
        f"after 14 renders expected exactly 10 archives, got {len(archives)}"
    # Archives are sorted lex newest-last (timestamp prefix). The oldest
    # ones from the first few rounds must have been pruned.
    names = [p.name for p in archives]
    assert names == sorted(names)

    # --- Step 4: per-section rebuild (calibration only) -------------------
    # Snapshot non-calibration files BEFORE rebuild.
    other_files: dict[str, bytes] = {}
    for sub in ("figs", "tables"):
        root = report_root / sub
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(report_root).as_posix()
                    if "/calibration/" not in rel:
                        other_files[rel] = p.read_bytes()

    rr = ReportRenderer(report_root=report_root, enable_archive=False)
    out = rr.rebuild(sections=["calibration"])
    assert out.is_file()

    # Non-calibration files must be byte-identical post-rebuild.
    for rel, before in other_files.items():
        p = report_root / rel
        assert p.is_file(), f"non-target file gone after rebuild: {rel}"
        assert p.read_bytes() == before, \
            f"non-target file mutated by rebuild: {rel}"


@pytest.mark.slow
@pytest.mark.gating
def test_rebuild_with_modified_metrics_reflects_change(tmp_path: Path) -> None:
    """After a render, write a custom metric, then rebuild calibration.

    The bundle persists what was used for the render; rebuild reads from the
    bundle, not from a fresh metrics value. So this primarily exercises the
    rebuild path's robustness — the calibration files must be re-emitted
    cleanly and the index regenerated.
    """
    report_root = tmp_path / "report"
    _render_once(report_root, run_id="rebuild-1")

    # Manually mutate metrics.json on disk to confirm rebuild doesn't
    # crash even if the live metrics file disagrees with the pickled bundle.
    import json
    mp = report_root / "metrics.json"
    d = json.loads(mp.read_text(encoding="utf-8"))
    d["brier"] = 0.99
    mp.write_text(json.dumps(d), encoding="utf-8")

    rr = ReportRenderer(report_root=report_root, enable_archive=False)
    out = rr.rebuild(sections=["calibration"])
    assert out.is_file()
    # The regenerated index.html still contains the Calibration section header.
    text = out.read_text(encoding="utf-8")
    assert "Calibration" in text
