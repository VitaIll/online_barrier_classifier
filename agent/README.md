# agent/

Agentic loop — **prompts and memory only**, no Python source. The loop runs
in-session inside Claude Code (per `RESEARCH/LOOP_DISCIPLINE.md`); this
folder holds the durable artifacts that survive across sessions.

```
agent/
├── prompts/      # role prompt templates (LITERATURE-SCOUT, CRITIC, …)
└── memory/       # per-session memory snapshots (curated)
```

The loop pattern (see `~/.claude/projects/.../memory/`):

- single Claude Code session, `/loop` dynamic mode
- `ScheduleWakeup` self-paces iterations
- commits go straight to `master` (one branch only)
- per-round work: pick one item from `RESEARCH/BACKLOG.md`, ship a measurable
  change with ≥1 chart, append `RESEARCH/LEDGER.md`

The agent's only entry into experiment running is the package CLI:

```bash
wagie experiment run experiments/<spec>.yaml
```

It must not bypass the protocol — no ad-hoc training, no bespoke charts.
