# Project State

This file is the current operational truth for the repository.

If a new AI agent starts, this is the first status file it should read after
the master roadmap.

## Updated At

2026-04-05

## Current Phase

Primary focus:

1. Phase 1 - Core Reliability
2. Phase 2 - Paper Trading Rehabilitation

Do not prioritize later phases until these are finished.

## Current Runtime Truth

- main operating mode: `robust_paper`
- live trading: disabled
- Azure runtime is reachable
- dashboard and APIs are reachable from the Azure VM public endpoint

## High-Confidence Findings

### 1. The bot is not dead; it is over-filtered

Recent Azure runtime data shows the system is running, but it is frequently
blocked by:

- `RISK_BLOCKED(normal)`
- `RISK_COMMITTEE:ModelDisagreement`
- `TRADE_BUDGET_TOO_SMALL`
- occasional stress-test rejection

### 2. The main bottleneck is paper-mode trade starvation

The system is producing too few trades to build a high-quality sample.

This is more important right now than adding advanced late-phase features.

### 3. Sizing is being compressed by stacked risk layers

Observed pattern:

- base Kelly position may be positive
- regime multipliers then reduce size
- Bayesian Kelly uncertainty then reduces it again
- final paper trade budget can fall below the minimum trade threshold
- result: `TRADE_BUDGET_TOO_SMALL`

### 4. Metrics lifecycle is noisy

Recent Azure logs showed:

- metrics bind warning on port `9090` because address is already in use

This does not appear to kill the bot, but it should be cleaned up.

## Current Known Validation State

Latest validation run on 2026-04-05:

- `ruff check wallet_manager tests` -> pass
- `mypy wallet_manager --ignore-missing-imports` -> pass
- `pytest tests -q` -> pass (`3155 passed`, `3 warnings`, `13m02s`)
- `pytest tests --cov=wallet_manager --cov-report=term --cov-fail-under=85 -q`
  -> fail (`82.70%` total coverage, target `85%`, `13m07s`)

Working rule for future sessions:

- do not rerun the full `3155`-test suite after every small edit
- use targeted module/subsystem tests during iteration
- rerun the canonical full-suite commands only after meaningful checkpoints or
  when a change is broad enough to justify it
- follow the tightened roadmap wording for Phase 1/Phase 2 measurement and
  declare phase completion only from repo-documented evidence

## Immediate Priorities

- [x] Re-run:
  - `ruff check wallet_manager tests`
  - `mypy wallet_manager --ignore-missing-imports`
  - `pytest tests -q`
  - `pytest tests --cov=wallet_manager --cov-report=term --cov-fail-under=85 -q`
- [x] Record exact results in `docs/HANDOFF.md`
- [x] Confirm whether Phase 1 is actually complete or not
- [ ] Raise total coverage from `82.70%` to at least `85%`
- [ ] Triage whether current coverage gap should be closed by new tests,
  dead-code removal, or justified coverage-scope cleanup for legacy/duplicate
  modules
- [ ] Keep iteration loops fast by using targeted validation first, then rerun
  the full canonical suite only at checkpoints
- [ ] Fix paper-mode bottlenecks in:
  - `wallet_manager/strategies/paper_engine.py`
  - `wallet_manager/strategies/deep_kelly_risk.py`
  - `wallet_manager/risk_intelligence.py`
  - `wallet_manager/strategies/bayesian_kelly.py`
- [ ] Re-check Azure runtime behavior after any runtime-affecting fix
- [ ] Clean up metrics/runtime lifecycle behavior around port `9090`

## Files Most Likely To Change Next

- `wallet_manager/strategies/paper_engine.py`
- `wallet_manager/strategies/deep_kelly_risk.py`
- `wallet_manager/risk_intelligence.py`
- `wallet_manager/strategies/bayesian_kelly.py`
- `wallet_manager/orchestrator.py`
- `wallet_manager/metrics_server.py`
- `wallet_manager/security/logging_filter.py`
- `tests/unit/test_task_factories_missing.py`
- `tests/unit/test_task_factories_missing2.py`
- `docs/HANDOFF.md`
- `deployment/AZURE_RUNTIME.md`

## Repo Documentation Structure

These are the only docs a new AI agent needs for continuity:

1. `docs/MASTER_ROADMAP.md`
2. `docs/PROJECT_STATE.md`
3. `docs/HANDOFF.md`
4. `deployment/AZURE_RUNTIME.md`
