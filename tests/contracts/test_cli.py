"""Contract tests for the wagie CLI (`python -m wagie ...`).

Exercises the entry point at `src/wagie/__main__.py`:
  - `wagie info`               → exits 0, prints version
  - `wagie experiment list`    → handles fresh empty dir gracefully
  - `wagie experiment show X`  → exits 1 on unknown run
  - `wagie experiment run S`   → exits 0 + produces an artifacts dir
  - bad subcommand             → non-zero exit
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
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
                "aci": {"alphas": [0.05, 0.10, 0.20], "gamma": 0.01,
                        "n_regimes": 3, "q_init": 0.5},
            },
            "strategy": {"kind": "pure_conformal", "alpha": 0.10},
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
    """Pointing list at a non-existent dir prints the documented placeholder."""
    target = tmp_path / "no_runs_here"
    cp = _run("experiment", "list", "--dir", str(target))
    assert cp.returncode == 0, f"stderr={cp.stderr!r}"
    assert "(no runs at " in cp.stdout
    assert str(target) in cp.stdout


def test_experiment_list_default_dir_when_missing(tmp_path: Path) -> None:
    """With no --dir, falls back to artifacts/runs (relative to CWD)."""
    cp = _run("experiment", "list", cwd=tmp_path)
    assert cp.returncode == 0
    assert "(no runs at " in cp.stdout


# -----------------------------------------------------------------
# `wagie experiment show <bad>`
# -----------------------------------------------------------------

def test_experiment_show_unknown_run_exits_one(tmp_path: Path) -> None:
    cp = _run("experiment", "show", "definitely_not_a_run",
              "--dir", str(tmp_path / "nope"))
    assert cp.returncode == 1
    assert "no such run" in (cp.stderr + cp.stdout)


# -----------------------------------------------------------------
# `wagie experiment run <good_spec>`
# -----------------------------------------------------------------

def test_experiment_run_good_spec_produces_artifacts(tmp_path: Path) -> None:
    parquet = _make_synthetic_parquet(tmp_path / "data.parquet")
    runs_dir = tmp_path / "runs"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_make_minimal_spec(parquet, runs_dir)))

    cp = _run("experiment", "run", str(spec_path), cwd=tmp_path)
    assert cp.returncode == 0, (
        f"stdout={cp.stdout!r}\nstderr={cp.stderr!r}"
    )

    # CLI prints the headline + out_dir
    assert "out_dir:" in cp.stdout
    assert "run_id=" in cp.stdout

    # An artifacts dir was created with the documented contents
    assert runs_dir.is_dir()
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1, f"expected 1 run dir, got {runs}"
    run_dir = runs[0]
    assert (run_dir / "spec.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()


def test_experiment_list_after_run_shows_entry(tmp_path: Path) -> None:
    parquet = _make_synthetic_parquet(tmp_path / "data.parquet")
    runs_dir = tmp_path / "runs"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_make_minimal_spec(parquet, runs_dir)))

    run_cp = _run("experiment", "run", str(spec_path), cwd=tmp_path)
    assert run_cp.returncode == 0

    list_cp = _run("experiment", "list", "--dir", str(runs_dir))
    assert list_cp.returncode == 0
    # Should show at least one run line — formatted as "name\tn=...\tsharpe=..."
    assert "n=" in list_cp.stdout
    assert "sharpe=" in list_cp.stdout
    assert "brier=" in list_cp.stdout


def test_experiment_show_existing_run_succeeds(tmp_path: Path) -> None:
    parquet = _make_synthetic_parquet(tmp_path / "data.parquet")
    runs_dir = tmp_path / "runs"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(_make_minimal_spec(parquet, runs_dir)))

    run_cp = _run("experiment", "run", str(spec_path), cwd=tmp_path)
    assert run_cp.returncode == 0
    runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    rid = runs[0].name

    show_cp = _run("experiment", "show", rid, "--dir", str(runs_dir))
    assert show_cp.returncode == 0
    # metrics.json content includes the trading block header
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
