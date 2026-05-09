# RESEARCH/

**Research files — separate from software.** Nothing in `src/wagie/` reads
this directory. Nothing here is imported or executed by the package.

## Layout

```
RESEARCH/
├── INDEX.md             # this file
├── CONSTITUTION.md      # invariants the loop must obey
├── AGENTS.md            # role contracts (LITERATURE-SCOUT, CRITIC, …)
├── LOOP_DISCIPLINE.md   # per-iteration structure + halt conditions
├── BACKLOG.md           # ordered queue of hypotheses
├── LEDGER.md            # append-only round log
├── HYPOTHESIS.md        # current round's working draft
├── KILL_LIST.md         # falsified hypotheses
├── HEALTH_OF_RESULTS.md # current state of art on this repo
├── REPORT.md            # canonical living dashboard
├── ROUND_TEMPLATE.md
├── literature/          # curated PDFs + INDEX.md
├── notes/               # free-form research notes
└── diagrams/            # per-round figures (auto-generated)
```

## Boundaries

- The package (`src/wagie/`) is the only thing that runs experiments.
- Research lives here — analysis, hypothesis statements, post-mortems.
- The loop reads from BACKLOG.md, writes to LEDGER.md, and spawns
  experiments via `wagie experiment run`. Nothing else.

## Literature

`literature/INDEX.md` is the local citation index — checked **before** any
web search. See [feedback memory](../agent/memory/) for the local-first rule.
