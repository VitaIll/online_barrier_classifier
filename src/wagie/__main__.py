"""wagie CLI — the SINGLE entry point.

Subcommands:
    wagie experiment run <spec.yaml>     # the canonical way
    wagie experiment list                 # list runs in artifacts/runs/
    wagie experiment show <run_id>        # echo metrics.json + report.md path
    wagie cv <spec.yaml>                  # cross-validation
    wagie data synth                      # write a deterministic synthetic
                                          # 1m parquet to data/synthetic/
    wagie info [--internal]               # version + public (or internal) surface

Nothing else in the repo provides a CLI. Custom scripts are forbidden — every
experiment goes through `wagie experiment run`.

Logging:
    Set ``WAGIE_LOG_FORMAT=json`` to emit one JSON record per line on stderr.
    See :mod:`wagie.observability` for the schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _cmd_experiment_run(args) -> int:
    from wagie.experiments import ExperimentProtocol, ExperimentSpec
    spec = ExperimentSpec.from_yaml(args.spec)
    # Light-touch fallback: if the spec points at the canonical synthetic
    # parquet but it isn't there yet, try the cleansed-data location next.
    data_path = Path(spec.wagie.data.parquet_path)
    if not data_path.is_file():
        candidates = [
            Path("data/synthetic/btcusdt_1m.parquet"),
            Path("data/cleansed_data/BTCUSDT/1m.parquet"),
        ]
        for cand in candidates:
            if cand.is_file():
                spec.wagie.data.parquet_path = str(cand)
                logging.info(f"data path fallback: {data_path} -> {cand}")
                break
    result = ExperimentProtocol().run(spec, spec_path=Path(args.spec))
    print(result.headline)
    print(f"out_dir: {result.out_dir}")
    if result.report_path:
        print(f"report: {result.report_path}")
    return 0


def _cmd_experiment_list(args) -> int:
    root = Path(args.dir or "artifacts/runs")
    if not root.is_dir():
        print(f"(no runs at {root})")
        return 0
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    for r in runs:
        m = r / "metrics.json"
        if m.exists():
            try:
                d = json.loads(m.read_text())
                t = d.get("trading", {})
                print(f"{r.name}\tn={t.get('n_trades', 0)}"
                      f"\tsharpe={t.get('sharpe', 0.0):+.3f}"
                      f"\tbrier={d.get('brier', 0.0):.5f}"
                      f"\tece={d.get('ece', 0.0):.5f}")
            except Exception:
                print(f"{r.name}\t(metrics unreadable)")
        else:
            print(f"{r.name}\t(no metrics.json)")
    return 0


def _cmd_experiment_show(args) -> int:
    root = Path(args.dir or "artifacts/runs")
    run = root / args.run_id
    if not run.is_dir():
        print(f"no such run: {run}", file=sys.stderr)
        return 1
    m = run / "metrics.json"
    if m.exists():
        print(m.read_text())
    rep = run / "report.md"
    if rep.exists():
        print(f"\n# report → {rep}")
    return 0


def _cmd_cv(args) -> int:
    """Force-run a spec in CV mode regardless of its `cv:` block."""
    from wagie.experiments import CVSpec, ExperimentProtocol, ExperimentSpec
    spec = ExperimentSpec.from_yaml(args.spec)
    spec.cv = CVSpec(
        enabled=True,
        n_folds=args.n_folds,
        n_test_folds=args.n_test_folds,
        embargo_size=args.embargo,
    )
    result = ExperimentProtocol().run(spec, spec_path=Path(args.spec))
    print(result.headline)
    print(f"out_dir: {result.out_dir}")
    if result.report_path:
        print(f"report: {result.report_path}")
    return 0


def _cmd_data_synth(args) -> int:
    """Write a deterministic synthetic 1m BTCUSDT parquet.

    Lets users run `wagie experiment run experiments/baseline.yaml` from a
    fresh clone with no cleansed_data on disk.
    """
    import numpy as np
    import polars as pl

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    n = int(args.n_minutes)
    sigma = float(args.sigma)

    log_ret = rng.normal(0.0, sigma, size=n)
    log_ret[0] = 0.0
    log_close = float(args.start_log_price) + np.cumsum(log_ret)
    close = np.exp(log_close)
    bar_range = np.abs(rng.normal(0.0, sigma * 1.2, size=n))
    high = close * (1.0 + bar_range)
    low = close * (1.0 - bar_range)
    open_ = np.r_[close[0], close[:-1]]
    volume = np.abs(rng.normal(100.0, 20.0, size=n))

    open_time = np.arange(n) * 60_000 + int(args.start_ms)
    close_time = open_time + 59_999

    # 4-decimal close per ask: round all OHLC to 4 decimals.
    close = np.round(close, 4)
    open_ = np.round(open_, 4)
    high = np.round(high, 4)
    low = np.round(low, 4)
    volume = np.round(volume, 4)

    df = pl.DataFrame({
        "open_time": open_time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "close_time": close_time,
        "quote_volume": np.round(volume * close, 4),
        "trades": np.full(n, 50, dtype=np.int64),
        "taker_buy_base": np.round(volume * 0.5, 4),
        "taker_buy_quote": np.round(volume * close * 0.5, 4),
        "segment_id": np.zeros(n, dtype=np.int64),
    })
    df.write_parquet(str(out_path))
    print(f"wrote {out_path}  rows={n}  bytes={out_path.stat().st_size}")
    return 0


def _cmd_info(args) -> int:
    import wagie
    print(f"wagie {wagie.__version__}")
    print(f"public types: {len(wagie.__all__)}")
    if getattr(args, "internal", False):
        # Internal surface: anything in the package namespace not on __all__.
        # See wagie.public vs wagie.internal in src/wagie/__init__.py docstring.
        public = set(wagie.__all__)
        internal = sorted(
            name for name in dir(wagie)
            if not name.startswith("_") and name not in public
        )
        print(f"internal symbols: {len(internal)}")
        for name in internal:
            print(f"  {name}")
    return 0


def main(argv=None) -> int:
    # Structured logging: WAGIE_LOG_FORMAT=json picks the JSON formatter.
    from wagie.observability import configure_logging
    configure_logging(level=logging.INFO)
    p = argparse.ArgumentParser(prog="wagie")
    sub = p.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("experiment", help="ExperimentProtocol commands")
    exp_sub = exp.add_subparsers(dest="exp_cmd", required=True)

    e_run = exp_sub.add_parser("run", help="Run a single experiment from a YAML spec.")
    e_run.add_argument("spec", type=str, help="path to experiment spec YAML")
    e_run.set_defaults(func=_cmd_experiment_run)

    e_list = exp_sub.add_parser("list", help="List runs in artifacts/runs.")
    e_list.add_argument("--dir", type=str, default=None,
                        help="run directory (default: artifacts/runs)")
    e_list.set_defaults(func=_cmd_experiment_list)

    e_show = exp_sub.add_parser("show", help="Show metrics + report path for a run.")
    e_show.add_argument("run_id", type=str)
    e_show.add_argument("--dir", type=str, default=None)
    e_show.set_defaults(func=_cmd_experiment_show)

    cv_p = sub.add_parser("cv", help="Run cross-validation over an experiment spec.")
    cv_p.add_argument("spec", type=str)
    cv_p.add_argument("--n-folds", type=int, default=10)
    cv_p.add_argument("--n-test-folds", type=int, default=2)
    cv_p.add_argument("--embargo", type=int, default=5)
    cv_p.set_defaults(func=_cmd_cv)

    data = sub.add_parser("data", help="Data utilities (synth fixtures).")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    d_synth = data_sub.add_parser(
        "synth",
        help="Write a deterministic synthetic 1m BTCUSDT parquet.",
    )
    d_synth.add_argument(
        "--out", type=str, default="data/synthetic/btcusdt_1m.parquet",
        help="destination path (default data/synthetic/btcusdt_1m.parquet)",
    )
    d_synth.add_argument(
        "--n-minutes", dest="n_minutes", type=int, default=129_600,
        help="number of 1-minute bars (default: ~3 months = 129600)",
    )
    d_synth.add_argument("--sigma", type=float, default=0.0008,
                         help="per-bar log-return stddev (default 0.0008)")
    d_synth.add_argument("--seed", type=int, default=42)
    d_synth.add_argument("--start-log-price", dest="start_log_price",
                         type=float, default=10.0,
                         help="initial log-price (default 10.0 ≈ $22026)")
    d_synth.add_argument("--start-ms", dest="start_ms",
                         type=int, default=1_700_000_000_000,
                         help="initial open_time in ms (default 2023-11-14)")
    d_synth.set_defaults(func=_cmd_data_synth)

    info_p = sub.add_parser("info", help="Print wagie version + public surface.")
    info_p.add_argument(
        "--internal", action="store_true",
        help="Also list internal symbols (not in wagie.__all__).",
    )
    info_p.set_defaults(func=_cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
