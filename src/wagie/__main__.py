"""wagie CLI — the SINGLE entry point.

Subcommands:
    wagie experiment run <spec.yaml>     # the canonical way
    wagie experiment list                 # list runs in artifacts/runs/
    wagie experiment show <run_id>        # echo metrics.json + report.md path
    wagie cv <spec.yaml>                  # cross-validation
    wagie info                            # version + public surface

Nothing else in the repo provides a CLI. Custom scripts are forbidden — every
experiment goes through `wagie experiment run`.
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


def _cmd_info(args) -> int:
    import wagie
    print(f"wagie {wagie.__version__}")
    print(f"public types: {len(wagie.__all__)}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
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

    info_p = sub.add_parser("info", help="Print wagie version + public surface.")
    info_p.set_defaults(func=_cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
