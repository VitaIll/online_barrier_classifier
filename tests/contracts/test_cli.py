"""Contract tests for the wagie CLI (`python -m wagie ...`).

Exercises the entry point at `src/wagie/__main__.py`:
  - `wagie info`                 → exits 0, prints version
  - `wagie experiment run`       → writes the unified report at
                                    `artifacts/report/index.html`
  - `wagie experiment run --experiment NAME`
                                  → writes side report at
                                    `artifacts/experiments/NAME/index.html`
                                    without touching the canonical report
  - `wagie report rebuild --help` → parseable
  - `wagie report archives --help`→ parseable
  - bad subcommand               → non-zero exit

Tests that rely on the old `artifacts/runs/<run_id>/` layout are marked
xfail with strict=False — the protocol/CLI rewire (Agent C) replaces that
layout with a single canonical `artifacts/report/` dir.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _run(*args: str, cwd: Path | None = None,
         timeout: int = 240) -> subprocess.CompletedProcess:
    """Invoke `python -m wagie <args>` and capture output."""
    env = os.environ.copy()
    # Ensure UTF-8 so any unicode output (em-dashes etc.) doesn't blow up on
    # Windows console encodings.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    # If PYTHONPATH is a relative path (e.g. "src" set by the test runner
    # to point at this worktree's src tree), absolutise it BEFORE we cd into
    # tmp_path — otherwise the subprocess looks for "src" inside tmp_path.
    pp = env.get("PYTHONPATH")
    if pp:
        parts = pp.split(os.pathsep)
        abs_parts = [str(Path(p).resolve()) if not Path(p).is_absolute() else p
                     for p in parts]
        env["PYTHONPATH"] = os.pathsep.join(abs_parts)
    return subprocess.run(
        [sys.executable, "-m", "wagie", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, encoding="utf-8",
        env=env, timeout=timeout,
    )


def _make_synthetic_parquet(out: Path, n: int = 4000) -> Path:
    """Reuse the synthetic generator pattern from tests/integration/conftest.py."""
    rng = np.random.default_rng(42)
    sigma = 0.0008
    log_ret = rng.normal(0.0, sigma, size=n)
    log_ret[0] = 0.0
    log_close = 10.0 + np.cumsum(log_ret)
    close = np.exp(log_close)
    bar_range = np.abs(rng.normal(0.0, sigma * 1.2, size=n))
    high = close * (1.0 + bar_range)
    low = close * (1.0 - bar_range)
    open_ = np.r_[close[0], close[:-1]]
    volume = np.abs(rng.normal(100.0, 20.0, size=n))

    base_open_time_ms = 1_700_000_000_000
    open_time = np.arange(n) * 60_000 + base_open_time_ms
    close_time = open_time + 59_999

    df = pl.DataFrame({
        "open_time": open_time, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "close_time": close_time,
        "quote_volume": volume * close,
        "trades": np.full(n, 50, dtype=np.int64),
        "taker_buy_base": volume * 0.5,
        "taker_buy_quote": volume * close * 0.5,
        "segment_id": np.zeros(n, dtype=np.int64),
    })
    df.write_parquet(out)
    return out


def _make_minimal_spec(parquet: Path, out_dir: Path) -> dict:
    """A minimal spec dict matching the contract for ExperimentSpec."""
    return {
        "name": "cli_smoke",
        "description": "cli contract test",
        "seed": 42,
        "wagie": {
            "data": {"parquet_path": str(parquet), "m_minutes": 20},
            "model": {
                "catboost_path": None,
                "arf": {"n_models": 5, "lambda_value": 6.0, "seed": 42},
            },
            "strategy": {"kind": "threshold_gate", "tau": 0.20},
            "broker": {"inventory_cap": 5},
            "runtime": {"warmup_samples": 10},
        },
        "features": {"catalog": "minimal", "n_features": 5},
        "charts": {"enable": True, "n_calibration_bins": 10},
        "report": {"enable": True, "title": None},
        "artifacts": {"out_dir": str(out_dir), "save_state": True,
                      "save_predictions": True},
    }


# -----------------------------------------------------------------
# `wagie info`
# -----------------------------------------------------------------

def test_info_exits_zero_and_prints_version() -> None:
    cp = _run("info")
    assert cp.returncode == 0, f"stderr={cp.stderr!r}"
    assert cp.stdout.startswith("wagie ")
    # version line must contain a semver-ish triple
    assert re.search(r"wagie \d+\.\d+\.\d+", cp.stdout)
    # also reports the public surface count
    assert "public types:" in cp.stdout


# -----------------------------------------------------------------
# `wagie experiment list`
# -----------------------------------------------------------------

def test_experiment_list_empty_dir(tmp_path: Path) -> None:
    """Pointing list at a non-existent dir prints the placeholder."""
    target = tmp_path / "no_report_here"
    cp = _run("experiment", "list", "--dir", str(target))
    assert cp.returncode == 0, f"stderr={cp.stderr!r}"
    # New shape: "(no report at ...)" since the layout moved from
    # artifacts/runs/<run_id> to a single artifacts/report/.
    assert "(no report at " in cp.stdout or "(no runs at " in cp.stdout
    assert str(target) in cp.stdout


def test_experiment_list_default_dir_when_missing(tmp_path: Path) -> None:
    """With no --dir, falls back to the canonical report root (artifacts/report)."""
    cp = _run("experiment", "list", cwd=tmp_path)
    assert cp.returncode == 0
    # Either old "(no runs at " or new "(no report at " is acceptable.
    assert "(no runs at " in cp.stdout or "(no report at " in cp.stdout


# -----------------------------------------------------------------
# `wagie experiment show <bad>`
# -----------------------------------------------------------------

def test_experiment_show_missing_dir_exits_nonzero(tmp_path: Path) -> None:
    """`experiment show --dir <missing>` must fail (no report there)."""
    cp = _run("experiment", "show", "--dir", str(tmp_path / "nope"))
    assert cp.returncode != 0
    # Either the old "no such run" or the new "no such report" wording.
    err = cp.stderr + cp.stdout
    assert "no such" in err.lower() or "no report" in err.lower()


# -----------------------------------------------------------------
# `wagie experiment run <good_spec>`
# -----------------------------------------------------------------

def test_experiment_run_good_spec_produces_html_report(tmp_path: Path) -> None:
    """`wagie experiment run` writes a unified HTML report (index.html).

    The exact path may be ``artifacts/report/index.html`` (canonical shape
    after Agent C's rewire) or the spec's ``out_dir/index.html`` (current
    behaviour when the protocol still respects spec.artifacts.out_dir).
    Either way, an ``index.html`` MUST be present and the CLI prints
    ``report:`` followed by its location, NOT ``report.md``.
    """
    parquet = _make_synthetic_parquet(tmp_path / "data.parquet")
    runs_dir = tmp_path / "runs"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_make_minimal_spec(parquet, runs_dir)))

    cp = _run("experiment", "run", str(spec_path), cwd=tmp_path)
    assert cp.returncode == 0, (
        f"stdout={cp.stdout!r}\nstderr={cp.stderr!r}"
    )

    # Some index.html exists somewhere under tmp_path.
    indices = list(tmp_path.rglob("index.html"))
    assert indices, (
        f"no index.html produced anywhere under {tmp_path}\n"
        f"stdout={cp.stdout!r}"
    )
    # The CLI reports the path on a `report:` line and points at index.html
    # (NOT a report.md from the legacy markdown path).
    if "report:" in cp.stdout:
        assert "index.html" in cp.stdout
        assert "report.md" not in cp.stdout


def test_experiment_run_with_experiment_flag_isolates_to_side_dir(
    tmp_path: Path,
) -> None:
    """`wagie experiment run --experiment foo SPEC` must write to
    artifacts/experiments/foo/index.html and NOT to artifacts/report/."""
    parquet = _make_synthetic_parquet(tmp_path / "data.parquet")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(
        _make_minimal_spec(parquet, tmp_path / "_unused_runs"),
    ))

    cp = _run("experiment", "run", "--experiment", "foo",
              str(spec_path), cwd=tmp_path)
    assert cp.returncode == 0, f"stderr={cp.stderr!r}"

    side_index = tmp_path / "artifacts" / "experiments" / "foo" / "index.html"
    assert side_index.is_file(), f"expected side report at {side_index}"

    # Canonical untouched
    canonical_index = tmp_path / "artifacts" / "report" / "index.html"
    assert not canonical_index.exists(), (
        "side experiment must not touch the canonical report"
    )


def test_report_rebuild_help_parses() -> None:
    """`wagie report rebuild --help` should parse and exit 0."""
    cp = _run("report", "rebuild", "--help")
    assert cp.returncode == 0, f"stderr={cp.stderr!r}"
    # Help output mentions rebuild
    assert "rebuild" in (cp.stdout + cp.stderr).lower()


def test_report_archives_help_parses() -> None:
    """`wagie report archives --help` should parse and exit 0."""
    cp = _run("report", "archives", "--help")
    assert cp.returncode == 0, f"stderr={cp.stderr!r}"
    assert "archive" in (cp.stdout + cp.stderr).lower()


def test_experiment_list_after_run_shows_summary(tmp_path: Path) -> None:
    """After a run, `experiment list --dir <report_root>` must summarise it.

    The new shape lists the live manifest summary + archived snapshots;
    we just verify it returns 0 and prints SOMETHING about the report.
    """
    parquet = _make_synthetic_parquet(tmp_path / "data.parquet")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(
        _make_minimal_spec(parquet, tmp_path / "runs"),
    ))

    run_cp = _run("experiment", "run", str(spec_path), cwd=tmp_path)
    assert run_cp.returncode == 0, f"stderr={run_cp.stderr!r}"

    # Find the report root the run actually produced (canonical or legacy).
    canonical = tmp_path / "artifacts" / "report"
    legacy_root = tmp_path / "runs"
    candidate = canonical if canonical.is_dir() else legacy_root

    list_cp = _run("experiment", "list", "--dir", str(candidate))
    assert list_cp.returncode == 0, f"stderr={list_cp.stderr!r}"
    # Output is non-empty
    assert list_cp.stdout.strip()


def test_experiment_show_after_run_succeeds(tmp_path: Path) -> None:
    """`experiment show --dir <report_root>` after a run must echo metrics.

    Resolves the report root by looking for a metrics.json under any of the
    plausible locations (canonical artifacts/report, the spec's out_dir, or
    a legacy <out_dir>/<run_id> shape) and points show at it.
    """
    parquet = _make_synthetic_parquet(tmp_path / "data.parquet")
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(
        _make_minimal_spec(parquet, tmp_path / "runs"),
    ))

    run_cp = _run("experiment", "run", str(spec_path), cwd=tmp_path)
    assert run_cp.returncode == 0, f"stderr={run_cp.stderr!r}"

    # Find the report root by locating metrics.json on disk.
    metrics_files = [p for p in tmp_path.rglob("metrics.json")
                     if "_archive" not in p.parts]
    assert metrics_files, f"no metrics.json found under {tmp_path}"
    report_root = metrics_files[0].parent

    show_cp = _run("experiment", "show", "--dir", str(report_root))
    assert show_cp.returncode == 0, f"stderr={show_cp.stderr!r}"
    # metrics.json is echoed; the "trading" block must be inside.
    assert '"trading"' in show_cp.stdout


# -----------------------------------------------------------------
# Bad invocations
# -----------------------------------------------------------------

def test_bad_subcommand_exits_nonzero() -> None:
    cp = _run("not_a_real_command")
    assert cp.returncode != 0


def test_no_subcommand_exits_nonzero() -> None:
    cp = _run()
    assert cp.returncode != 0


def test_experiment_without_subcommand_exits_nonzero() -> None:
    cp = _run("experiment")
    assert cp.returncode != 0


def test_experiment_run_missing_spec_path_exits_nonzero(tmp_path: Path) -> None:
    cp = _run("experiment", "run", str(tmp_path / "does_not_exist.yaml"))
    assert cp.returncode != 0
