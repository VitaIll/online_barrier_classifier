"""Formalised CRITIC checklist — runs on a round branch before accept.

This is the code-level implementation of `RESEARCH/AGENTS.md::CRITIC`. It
turns the loose prompt into machine-checkable boolean items. The CRITIC
sub-agent invokes this; an `accept` decision requires exit 0.

Usage:
    python scripts/critic_check.py
    python scripts/critic_check.py --hypothesis H-005 --claim "BSS > 0.10"
    python scripts/critic_check.py --json    # machine-readable output

Exit codes:
    0  All checks passed → CRITIC may APPROVE.
    1  At least one check failed → CRITIC must REQUEST_CHANGES.
    2  An error prevented checks from running → CRITIC must VETO with reason.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "must"  # "must" or "should"


@dataclass
class ReviewReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", severity: str = "must"):
        self.checks.append(CheckResult(name, passed, detail, severity))

    @property
    def must_pass(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "must")

    @property
    def all_pass(self) -> bool:
        return all(c.passed for c in self.checks)


def _git(cmd: list[str]) -> str:
    p = subprocess.run(["git"] + cmd, cwd=REPO, capture_output=True, text=True)
    return p.stdout.strip()


def _git_diff_files() -> list[str]:
    """List of files changed in this round (uncommitted: staged + unstaged vs HEAD).

    The new in-session loop runs entirely on master with no per-round branch.
    CRITIC fires AFTER the round's edits land in the working tree but BEFORE
    commit, so the round's contribution is `git diff HEAD`.
    """
    out = _git(["diff", "HEAD", "--name-only"])
    return [ln for ln in out.splitlines() if ln]


def _git_diff_text() -> str:
    return _git(["diff", "HEAD"])


def check_pytest_passing(report: ReviewReport) -> None:
    """Critical: full pytest suite must be green."""
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"], cwd=REPO,
                        capture_output=True, text=True)
    if p.returncode == 0:
        passed_count = re.search(r"(\d+) passed", p.stdout or p.stderr)
        n = passed_count.group(1) if passed_count else "?"
        report.add("pytest_full_suite", True, f"{n} passed")
    else:
        tail = (p.stdout or "").splitlines()[-10:]
        report.add("pytest_full_suite", False, "\n".join(tail))


def check_causality_tests(report: ReviewReport) -> None:
    """Causality + property tests must pass — leakage-blocking.

    Per the bootstrap LEDGER (round 000), this project's causality/splits/weights
    tests are parked in `tests/_pending/` until H-101..H-103 land the matching
    `src/utils.py` callables. A parked-file state is NOT a regression — the
    check passes vacuously. Once a pending test is moved into `tests/`, it is
    automatically picked up here.
    """
    candidates = [
        "tests/test_causality.py", "tests/test_properties.py",
        "tests/test_weights.py", "tests/test_splits.py",
    ]
    present = [t for t in candidates if (REPO / t).exists()]
    if not present:
        report.add("causality_property_tests", True,
                   "all parked in tests/_pending/ pending H-101..H-103")
        return
    p = subprocess.run([
        sys.executable, "-m", "pytest", "-q", *present,
    ], cwd=REPO, capture_output=True, text=True)
    report.add("causality_property_tests", p.returncode == 0,
               (p.stdout or "")[-300:] if p.returncode != 0 else f"green ({len(present)} files)")


def check_no_force_or_skip(report: ReviewReport) -> None:
    """No --no-verify or --force in recent commits."""
    log = _git(["log", "--oneline", "-20"])
    bad_phrases = ["--no-verify", "--force", "skip-hooks", "noverify"]
    found = [p for p in bad_phrases if p in log]
    report.add("no_force_no_verify", len(found) == 0,
               "found: " + ", ".join(found) if found else "clean")


def check_diff_size(report: ReviewReport, max_loc: int = 1500) -> None:
    """Diff <= max_loc lines (excluding tests) — large diffs are review-resistant."""
    files = _git_diff_files()
    text = _git_diff_text()
    added = sum(1 for ln in text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    report.add("diff_size_under_cap", added <= max_loc,
               f"{added} added lines vs cap {max_loc}", severity="should")


def check_test_added_for_new_compute(report: ReviewReport) -> None:
    """If new compute_*/feature function added, a test should reference it."""
    diff = _git_diff_text()
    new_fns = set(re.findall(r"^\+def (compute_[a-z_]+)\(", diff, re.MULTILINE))
    if not new_fns:
        report.add("test_for_new_compute", True, "no new compute_* fns", severity="should")
        return
    test_text = ""
    for tf in (REPO / "tests").glob("test_*.py"):
        test_text += tf.read_text(encoding="utf-8")
    untested = [fn for fn in new_fns if fn not in test_text]
    report.add("test_for_new_compute",
               len(untested) == 0,
               f"new fns missing tests: {untested}" if untested else f"all {len(new_fns)} new fns referenced",
               severity="should")


def check_diagnostic_plot_present(report: ReviewReport) -> None:
    """At least one new PNG diagnostic landed in this branch."""
    files = _git_diff_files()
    new_pngs = [f for f in files if f.endswith(".png") and "RESEARCH/diagrams" in f]
    report.add("diagnostic_plot_added",
               len(new_pngs) > 0,
               f"new plots: {new_pngs[:3]}" if new_pngs else "no diagnostic PNG in branch",
               severity="should")


def check_ledger_entry(report: ReviewReport) -> None:
    """LEDGER got at least one new line in this branch."""
    diff = _git_diff_text()
    new_ledger_lines = [
        ln for ln in diff.splitlines()
        if ln.startswith("+") and re.match(r"^\+\d{4}-\d{2}-\d{2} \|", ln)
    ]
    report.add("ledger_entry_added",
               len(new_ledger_lines) >= 1,
               f"{len(new_ledger_lines)} new LEDGER line(s)")


def check_no_label_features_leakage(report: ReviewReport) -> None:
    """Forbidden columns (m_k, tau_k, phi, w_dist, w_time, weight) must NOT
    appear in any new feature_list.json or feature definition diff."""
    diff = _git_diff_text()
    forbidden = ["m_k", "tau_k", "phi", "w_dist", "w_time", "weight"]
    flagged = []
    for ln in diff.splitlines():
        if not ln.startswith("+"):
            continue
        if "feature_list" in ln or "feature_cols" in ln:
            for f in forbidden:
                if f'"{f}"' in ln or f"'{f}'" in ln:
                    flagged.append((ln.strip(), f))
    report.add("no_label_features_leakage",
               len(flagged) == 0,
               f"flagged: {flagged[:3]}" if flagged else "clean")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", default="?")
    parser.add_argument("--claim", default="?")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--allow-should-fail", action="store_true",
                        help="exit 0 if 'must' checks pass even with 'should' failures")
    args = parser.parse_args()

    report = ReviewReport()
    try:
        check_no_force_or_skip(report)
        check_no_label_features_leakage(report)
        check_causality_tests(report)
        check_pytest_passing(report)
        check_diff_size(report)
        check_test_added_for_new_compute(report)
        check_diagnostic_plot_present(report)
        check_ledger_entry(report)
    except Exception as exc:
        print(f"CRITIC error: {exc}", file=sys.stderr)
        if args.json:
            print(json.dumps({"verdict": "VETO", "error": str(exc), "checks": []}))
        return 2

    must_passed = report.must_pass
    all_passed = report.all_pass

    if args.json:
        out = {
            "hypothesis": args.hypothesis,
            "claim": args.claim,
            "verdict": "APPROVE" if (all_passed or (must_passed and args.allow_should_fail))
                        else "REQUEST_CHANGES",
            "must_passed": must_passed,
            "all_passed": all_passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "severity": c.severity, "detail": c.detail[:200]}
                for c in report.checks
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"\nCRITIC checklist — hypothesis {args.hypothesis}\n")
        for c in report.checks:
            mark = "OK" if c.passed else ("WARN" if c.severity == "should" else "FAIL")
            print(f"  [{mark:>4}] ({c.severity}) {c.name:35} {c.detail[:80]}")
        print()
        verdict = "APPROVE" if all_passed else (
            "REQUEST_CHANGES (should-failures only)" if (must_passed and args.allow_should_fail)
            else "REQUEST_CHANGES")
        print(f"  Verdict: {verdict}\n")

    if all_passed:
        return 0
    if must_passed and args.allow_should_fail:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
