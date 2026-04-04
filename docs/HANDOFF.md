# Handoff

This file exists for one purpose:

When an AI session ends, hits a limit, or gets interrupted, the next AI agent
must be able to continue from the exact same point with minimal guessing.

## Required Usage

Every AI agent should:

1. read this file before making changes
2. update the `Active Handoff` section before ending the session
3. update it before launching long-running commands if interruption is possible
4. update `deployment/AZURE_RUNTIME.md` if Azure/runtime facts change

## Read Order For A New AI Agent

1. `docs/MASTER_ROADMAP.md`
2. `docs/PROJECT_STATE.md`
3. `docs/HANDOFF.md`
4. `deployment/AZURE_RUNTIME.md`
5. `deployment/OPERATOR_RUNBOOK.md`

## Active Handoff

### Updated At

2026-04-05 01:12:05 +03:00

### Current Objective

Finish the highest-priority work in order:

1. Phase 1 - Core Reliability
2. Phase 2 - Paper Trading Rehabilitation

### Last Completed Work

- revalidated the full Phase 1 command set after cleanup/doc restructuring
- reduced `ruff` findings from 548 to 0 and made test lint policy explicit for non-critical test-only style rules
- hardened `wallet_manager/security/logging_filter.py` so mocked/non-string `log_level` values fall back safely
- fixed full-suite-only test isolation issues in:
  - `tests/unit/test_task_factories_missing.py`
  - `tests/unit/test_task_factories_missing2.py`
- confirmed:
  - `ruff check wallet_manager tests` passes
  - `mypy wallet_manager --ignore-missing-imports` passes
  - `pytest tests -q` passes (`3155 passed`, `3 warnings`, about `13m`)
- confirmed Phase 1 is still incomplete because coverage is `82.70%` and the gate requires `>= 85%`
- updated repo docs so future sessions use targeted validation during iteration
  and reserve the full-suite sweep for checkpoints/handoff/phase-exit
- tightened `docs/MASTER_ROADMAP.md` so:
  - roadmap maintenance cannot weaken gates
  - local iteration versus checkpoint validation is explicit
  - Phase 2 now requires a documented baseline-vs-post-change comparison
- stopped stale local `pytest` / `coverage` `python.exe` processes that were
  likely holding SQLite files open during archive attempts
- initialized a local git repository on branch `main`
- expanded `.gitignore` for cache/editor/coverage artifacts to make GitHub
  transfer cleaner
- did not create a commit yet because local git identity (`user.name`,
  `user.email`) is not configured in this environment

### Exact Continuation Point

The next AI agent should:

1. open `docs/PROJECT_STATE.md`
2. read the tightened wording in `docs/MASTER_ROADMAP.md` before making
   phase-closure claims
3. treat Phase 1 as blocked only by coverage, not by lint/type/test failures
4. use targeted tests for the touched module while iterating; do not rerun the
   full `3155`-test suite after every small change
5. use the latest full-suite results in this file unless new code changes make
   them stale
6. raise total coverage from `82.70%` to `>= 85%`
7. rerun the full canonical suite only at a checkpoint after meaningful
   coverage-focused changes
8. only begin Phase 2 bottleneck fixes after the coverage gate passes
9. if GitHub transfer is next, set git identity, review whether `sync.zip`
   should stay out of the repo, then stage/commit/add remote/push

### Long-Running Or Interrupted Work

Use this section whenever a command may outlive the session or hit a limit.

Current status:

- no known long-running local command is still in progress
- no known in-progress file edit is waiting to be completed
- partial findings so far:
  - full validation completed in this session
  - Phase 1 gate status is now blocked only by coverage (`82.70% < 85%`)
  - local archive lock issue should be reduced because stale pytest/coverage
    processes were terminated

If a future session is interrupted, it must update this block in this format:

- command started:
- started at:
- purpose:
- expected output file or effect:
- safe to rerun: yes/no
- next check command:
- partial findings so far:

### Last Known Important Runtime Facts

- Azure runtime reachable
- mode: `robust_paper`
- live trading disabled
- dominant no-trade reasons:
  - `RISK_BLOCKED(normal)`
  - `RISK_COMMITTEE:ModelDisagreement`
  - `TRADE_BUDGET_TOO_SMALL`

### Last Commands Run

- `ruff check wallet_manager tests` -> pass
- `mypy wallet_manager --ignore-missing-imports` -> pass
- `pytest tests -q` -> pass (`3155 passed`, `3 warnings`, `13m02s`)
- `pytest tests --cov=wallet_manager --cov-report=term --cov-fail-under=85 -q`
  -> fail (`82.70%`, target `85%`, `13m07s`)
- Azure docs/runtime facts were not changed in this session
- local git repo initialized with `git init -b main`
- local git identity is still unset

### Known Blockers

- Phase 1 coverage gate still fails at `82.70%`
- full-suite runtime is long (`~13 minutes` for pytest, `~13 minutes` for coverage)
- remaining warnings to evaluate later:
  - `aiosqlite` thread warning in `tests/unit/test_scheduler_analytics_coverage.py`
  - un-awaited coroutine warning in `tests/unit/test_task_factories_missing.py`
  - un-awaited coroutine warning in `tests/unit/test_telegram_commands_coverage.py`
- GitHub push is not yet possible until a remote and git identity are set
- paper mode still appears sample-starved
- stacked risk/sizing layers compress paper budgets too aggressively

### First Command To Run

```powershell
pytest tests --cov=wallet_manager --cov-report=term --cov-fail-under=85 -q
```

Only rerun this after making additional coverage-focused changes.

## Update Checklist Before Ending A Session

- [ ] update `Last Completed Work`
- [ ] update `Exact Continuation Point`
- [ ] update `Long-Running Or Interrupted Work` if relevant
- [ ] record important commands and results
- [ ] update `deployment/AZURE_RUNTIME.md` if runtime facts changed
