"""wagie CLI — the SINGLE entry point.

Subcommands:
    wagie experiment run <spec.yaml>      # the canonical way
        --experiment NAME                  # side run -> artifacts/experiments/<name>/
        --no-plotly                        # force matplotlib-only report
    wagie experiment list                  # archived snapshots + live summary
        --dir DIR                          # custom report root (default: artifacts/report)
    wagie experiment show                  # echo metrics.json + report path
        --experiment NAME                  # look under artifacts/experiments/<name>/

    wagie report rebuild                   # rerun selected sections from cached bundle
        --section NAME ...                 # repeatable; default: all
        --no-plotly                        # force static images on rebuild
        --dir DIR                          # custom report root
    wagie report archives                  # list zips in <dir>/_archive/
        --dir DIR
    wagie report show                      # alias for `experiment show`
        --dir DIR

    wagie cv <spec.yaml>                   # cross-validation
    wagie data synth                       # write a deterministic synthetic 1m parquet
    wagie info [--internal]                # version + public (or internal) surface

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
from typing import Optional


_DEFAULT_REPORT_ROOT = "artifacts/report"


# =============================================================================
# experiment run
# =============================================================================

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

    # --no-plotly overrides spec.report.use_plotly.
    if getattr(args, "no_plotly", False):
        spec.report.use_plotly = False

    experiment_name = getattr(args, "experiment", None)
    result = ExperimentProtocol().run(
        spec, spec_path=Path(args.spec), experiment_name=experiment_name,
    )
    print(result.headline)
    print(f"out_dir: {result.out_dir}")
    if result.report_path:
        print(f"report: {result.report_path}")
    return 0


# =============================================================================
# experiment list / report archives — both walk <dir>/_archive/*.zip
# =============================================================================

def _print_live_manifest_summary(report_root: Path) -> None:
    """Print the LIVE manifest + metrics summary if present."""
    manifest = report_root / "manifest.json"
    metrics = report_root / "metrics.json"
    if not manifest.is_file():
        print(f"(no live manifest at {manifest})")
        return
    try:
        d = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"(live manifest unreadable: {e})")
        return
    rm = d.get("run_meta") or {}
    accepted = bool(rm.get("accepted", True))
    badge = "OK" if accepted else "BLOCKED"
    print(
        f"LIVE  [{badge}]  spec={rm.get('spec_name','?')}  "
        f"hash={rm.get('spec_hash','?')[:8]}  mode={rm.get('mode','?')}  "
        f"generated_at={d.get('generated_at_utc','?')}"
    )
    if metrics.is_file():
        try:
            m = json.loads(metrics.read_text(encoding="utf-8"))
            t = m.get("trading", {}) or {}

            def _fmt(v, spec):
                return format(v, spec) if isinstance(v, (int, float)) else "n/a"

            print(
                f"      n_trades={t.get('n_trades', 0)}  "
                f"sharpe={_fmt(t.get('sharpe'), '+.3f')}  "
                f"brier={_fmt(m.get('brier'), '.5f')}  "
                f"ece={_fmt(m.get('ece'), '.5f')}"
            )
        except (json.JSONDecodeError, OSError):
            pass


def _print_archive_list(report_root: Path, *, max_show: int = 10) -> None:
    """Print up to ``max_show`` newest archived snapshots (lex-sorted)."""
    from wagie.reporting import list_archives
    archives = list_archives(report_root)
    if not archives:
        print(f"(no archives at {report_root / '_archive'})")
        return
    # newest-last from list_archives, so show the last N reversed (newest first)
    print(f"archives ({len(archives)} total, showing newest {min(max_show, len(archives))}):")
    for p in reversed(archives[-max_show:]):
        try:
            size_kb = p.stat().st_size / 1024.0
        except OSError:
            size_kb = 0.0
        # filename pattern: YYYYMMDDTHHMMSSZ_<8hex>.zip — dump as-is.
        print(f"  {p.name}\t{size_kb:.1f} KB")


def _cmd_experiment_list(args) -> int:
    """List archived snapshots + the live manifest summary."""
    report_root = Path(args.dir or _DEFAULT_REPORT_ROOT)
    if not report_root.is_dir():
        print(f"(no report at {report_root})")
        return 0
    _print_live_manifest_summary(report_root)
    print()
    _print_archive_list(report_root, max_show=10)
    return 0


def _cmd_report_archives(args) -> int:
    """List archive zips under <dir>/_archive/."""
    report_root = Path(args.dir or _DEFAULT_REPORT_ROOT)
    if not report_root.is_dir():
        print(f"(no report at {report_root})")
        return 1
    _print_archive_list(report_root, max_show=10**6)
    return 0


# =============================================================================
# experiment show / report show — same body
# =============================================================================

def _cmd_experiment_show(args) -> int:
    """Print metrics.json summary + path to index.html for the canonical
    report. With ``--experiment NAME`` looks under
    ``artifacts/experiments/<name>/`` instead.
    """
    if getattr(args, "experiment", None):
        report_root = Path("artifacts/experiments") / args.experiment
    else:
        report_root = Path(args.dir or _DEFAULT_REPORT_ROOT)

    if not report_root.is_dir():
        print(f"no such report: {report_root}", file=sys.stderr)
        return 1
    metrics = report_root / "metrics.json"
    if metrics.is_file():
        print(metrics.read_text(encoding="utf-8"))
    else:
        print(f"(no metrics.json at {metrics})")
    index = report_root / "index.html"
    if index.is_file():
        print(f"\n# report -> {index}")
    else:
        print(f"(no index.html at {index})", file=sys.stderr)
    return 0


# =============================================================================
# report rebuild
# =============================================================================

def _flatten_sections(values) -> Optional[list[str]]:
    """Flatten the argparse value(s) for --section into a list[str].

    With ``action='append'`` + ``nargs='+'`` we get a list of lists; with
    plain ``action='append'`` we get a list of single strings. None means
    "rebuild all".
    """
    if not values:
        return None
    out: list[str] = []
    for v in values:
        if isinstance(v, (list, tuple)):
            out.extend(str(x) for x in v if str(x).strip())
        else:
            out.append(str(v))
    return [s for s in out if s.strip()] or None


def _cmd_report_rebuild(args) -> int:
    """Rebuild selected sections from the cached rebuild bundle."""
    from wagie.reporting import ReportRenderer
    report_root = Path(args.dir or _DEFAULT_REPORT_ROOT)
    if not report_root.is_dir():
        print(
            f"no report at {report_root} -- run `wagie experiment run "
            f"<spec.yaml>` first",
            file=sys.stderr,
        )
        return 1
    manifest = report_root / "manifest.json"
    bundle = report_root / "state" / "rebuild_bundle.pkl"
    if not manifest.is_file():
        print(
            f"no manifest at {manifest} -- run `wagie experiment run "
            f"<spec.yaml>` first",
            file=sys.stderr,
        )
        return 1
    if not bundle.is_file():
        print(
            f"no rebuild bundle at {bundle} -- the previous render did not "
            f"persist its state; rerun `wagie experiment run`",
            file=sys.stderr,
        )
        return 1

    sections = _flatten_sections(getattr(args, "section", None))

    renderer = ReportRenderer(report_root=report_root)
    try:
        index_path = renderer.rebuild(sections=sections, title=None)
    except FileNotFoundError as e:
        print(f"rebuild failed: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        print(f"rebuild failed: {e}", file=sys.stderr)
        return 1

    # If --no-plotly was passed, force a re-render of the index without
    # Plotly inclusion. The renderer already takes the bundle's flag, so
    # re-render with use_plotly=False by patching the bundle in-place.
    if getattr(args, "no_plotly", False):
        try:
            import pickle
            with bundle.open("rb") as f:
                rb = pickle.load(f)
            rb.use_plotly = False
            with bundle.open("wb") as f:
                pickle.dump(rb, f, protocol=4)
            # And rebuild again with the patched bundle.
            index_path = renderer.rebuild(sections=sections, title=None)
        except (OSError, pickle.PickleError) as e:
            print(
                f"warning: --no-plotly toggle failed ({e}); "
                f"report still rendered with prior plotly setting",
                file=sys.stderr,
            )

    targets = sections if sections else "ALL"
    print(f"rebuild OK  sections={targets}")
    print(f"report: {index_path}")
    return 0


# =============================================================================
# cv (force-CV runner)
# =============================================================================

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


# =============================================================================
# data synth (unchanged)
# =============================================================================

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


# =============================================================================
# info
# =============================================================================

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


# =============================================================================
# main
# =============================================================================

def main(argv=None) -> int:
    # Structured logging: WAGIE_LOG_FORMAT=json picks the JSON formatter.
    from wagie.observability import configure_logging
    configure_logging(level=logging.INFO)
    p = argparse.ArgumentParser(prog="wagie")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- experiment ---------------------------------------------------
    exp = sub.add_parser("experiment", help="ExperimentProtocol commands")
    exp_sub = exp.add_subparsers(dest="exp_cmd", required=True)

    e_run = exp_sub.add_parser(
        "run", help="Run a single experiment from a YAML spec.",
    )
    e_run.add_argument("spec", type=str, help="path to experiment spec YAML")
    e_run.add_argument(
        "--experiment", type=str, default=None,
        help="side experiment NAME -> writes to artifacts/experiments/<NAME>/ "
             "without touching the canonical report",
    )
    e_run.add_argument(
        "--no-plotly", dest="no_plotly", action="store_true",
        help="force matplotlib-only output (overrides report.use_plotly)",
    )
    e_run.set_defaults(func=_cmd_experiment_run)

    e_list = exp_sub.add_parser(
        "list",
        help="List archived snapshots in <dir>/_archive/ + the live manifest.",
    )
    e_list.add_argument(
        "--dir", type=str, default=None,
        help=f"report root (default: {_DEFAULT_REPORT_ROOT})",
    )
    e_list.set_defaults(func=_cmd_experiment_list)

    e_show = exp_sub.add_parser(
        "show", help="Print metrics.json + index.html path for the report.",
    )
    e_show.add_argument(
        "--dir", type=str, default=None,
        help=f"report root (default: {_DEFAULT_REPORT_ROOT})",
    )
    e_show.add_argument(
        "--experiment", type=str, default=None,
        help="side experiment NAME -> reads artifacts/experiments/<NAME>/",
    )
    e_show.set_defaults(func=_cmd_experiment_show)

    # ---- report -------------------------------------------------------
    rep = sub.add_parser(
        "report",
        help="Per-section report rebuild + archive inspection.",
    )
    rep_sub = rep.add_subparsers(dest="rep_cmd", required=True)

    r_rebuild = rep_sub.add_parser(
        "rebuild",
        help="Rerun selected report sections from the cached rebuild bundle.",
    )
    r_rebuild.add_argument(
        "--section", action="append", default=None,
        help="section name (repeatable; default: rebuild all sections)",
    )
    r_rebuild.add_argument(
        "--no-plotly", dest="no_plotly", action="store_true",
        help="force matplotlib-only output for the rebuilt sections",
    )
    r_rebuild.add_argument(
        "--dir", type=str, default=None,
        help=f"report root (default: {_DEFAULT_REPORT_ROOT})",
    )
    r_rebuild.set_defaults(func=_cmd_report_rebuild)

    r_archives = rep_sub.add_parser(
        "archives",
        help="List archived snapshots in <dir>/_archive/.",
    )
    r_archives.add_argument(
        "--dir", type=str, default=None,
        help=f"report root (default: {_DEFAULT_REPORT_ROOT})",
    )
    r_archives.set_defaults(func=_cmd_report_archives)

    r_show = rep_sub.add_parser(
        "show",
        help="Print metrics.json + index.html path for the report.",
    )
    r_show.add_argument(
        "--dir", type=str, default=None,
        help=f"report root (default: {_DEFAULT_REPORT_ROOT})",
    )
    r_show.add_argument(
        "--experiment", type=str, default=None,
        help="side experiment NAME -> reads artifacts/experiments/<NAME>/",
    )
    r_show.set_defaults(func=_cmd_experiment_show)

    # ---- cv -----------------------------------------------------------
    cv_p = sub.add_parser("cv", help="Run cross-validation over an experiment spec.")
    cv_p.add_argument("spec", type=str)
    cv_p.add_argument("--n-folds", type=int, default=10)
    cv_p.add_argument("--n-test-folds", type=int, default=2)
    cv_p.add_argument("--embargo", type=int, default=5)
    cv_p.set_defaults(func=_cmd_cv)

    # ---- data ---------------------------------------------------------
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
                         help="initial log-price (default 10.0 ~ $22026)")
    d_synth.add_argument("--start-ms", dest="start_ms",
                         type=int, default=1_700_000_000_000,
                         help="initial open_time in ms (default 2023-11-14)")
    d_synth.set_defaults(func=_cmd_data_synth)

    # ---- info ---------------------------------------------------------
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
