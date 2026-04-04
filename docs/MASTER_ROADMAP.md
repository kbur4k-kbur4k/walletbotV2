# Master Roadmap

This file is the final canonical roadmap for the repository.

If multiple AI agents, operators, or future maintainers work on this project,
they must treat this file as the single source of truth for product direction,
software architecture direction, sequencing, and completion gates.

Do not create alternate roadmap files.
Do not skip phases.
Do not optimize for novelty over measurable progress.

## 1. Program Definition

### 1.1 North Star

Build an institutional-grade systematic trading platform that:

- measures every decision
- explains every trade and skipped trade
- produces hypotheses and evaluates them with evidence
- separates alpha generation from risk and execution
- promotes and demotes behavior through controlled stages
- survives regime changes without collapsing into guesswork

The goal is not a "smart bot".

The goal is a durable research, risk, execution, and promotion machine.

### 1.2 What "Jim Simons Direction" Means Here

For this repository, moving toward a Jim Simons-style system means:

- better measurement, not more hype
- better research process, not more indicators
- better calibration, not louder confidence numbers
- better promotion control, not direct jumps into live trading
- better architecture, not more scattered modules

This project should become a software and research platform where:

- ideas are cheap to test
- weak ideas die quickly
- strong ideas are promoted slowly
- every production behavior is measurable
- every material change is reversible

### 1.3 Practical Truth

This roadmap does not claim that 90 days creates literal Renaissance-level
edge.

What 90 days should produce is:

- a reliable quant platform foundation
- a research engine that can discover and reject weak ideas
- a risk and execution stack that can support controlled live progression

That is the correct software target.

## 2. Non-Negotiables

These rules are binding for all work sessions.

### 2.1 Sequence Rule

Work strictly in phase order.

No later phase is allowed to outrank the current phase's exit criteria.

### 2.2 Evidence Rule

No major behavior change is complete without:

- code changes
- verification
- measurement
- updated repo-side documentation

Verification must be proportional to the change while work is in progress.

- during local iteration, run the smallest targeted test or validation set that
  can falsify the current change
- rerun the full canonical validation suite only at meaningful checkpoints:
  before phase-closure claims, before handoff, before merge/deploy decisions,
  or after broad cross-cutting changes
- do not require a full-repo test sweep after every small edit if a narrower
  validation step gives faster and sufficient feedback

### 2.3 Promotion Rule

No behavior goes directly from idea to live capital.

The default promotion path is:

- research
- backtest
- paper
- shadow_live
- tiny_live
- scaled_live

### 2.4 Architecture Rule

The codebase must converge toward clear platform layers:

- runtime core
- data layer
- signal layer
- risk layer
- execution layer
- research layer
- observability layer
- promotion layer

### 2.5 Paper Mode Rule

Paper mode is not a decorative demo.

Paper mode is a data-generation and decision-quality environment. It must:

- produce enough samples
- produce trustworthy skip reasons
- mimic runtime flow closely enough to support promotion decisions

### 2.6 Documentation Rule

Operational truth must live inside the repo, not only in chat history.

### 2.7 Iteration Rule

Development loops should stay fast while phase gates stay strict.

- use targeted validation while shaping a fix
- widen validation only when the narrower check passes
- reserve full-repo sweeps for checkpoints, not every tiny edit

### 2.8 Roadmap Maintenance Rule

This roadmap may be refined, but only in place and only to improve execution
quality.

Allowed changes:

- tighter definitions
- clearer sequencing
- more measurable gates
- better continuity rules

Disallowed changes:

- weakening the current phase gate because it is inconvenient
- changing phase order without repo-side evidence
- creating alternate roadmap files or shadow priorities

## 3. End-State Platform Architecture

The system should converge toward the following canonical architecture.

### 3.1 Runtime Core

Responsibilities:

- orchestrator lifecycle
- wallet task dispatch
- scheduling
- startup validation
- health/readiness
- mode transitions

Primary file zones:

- `wallet_manager/orchestrator.py`
- `wallet_manager/orchestrator_lifecycle.py`
- `wallet_manager/task_factories.py`
- `wallet_manager/config.py`
- `wallet_manager/scheduler.py`

### 3.2 Data Layer

Responsibilities:

- historical datasets
- tick storage
- feature persistence
- decision journals
- paper/live runtime audit data
- replayable research slices

Primary file zones:

- `wallet_manager/resilience/analytics.py`
- `wallet_manager/tick_collector.py`
- `scripts/data/*`
- `historical_data.db`
- `quant_analytics.db`
- `paper_trades.db`
- `wallet_analytics.db`

### 3.3 Signal Layer

Responsibilities:

- feature engineering
- model inference
- alpha scoring
- confidence estimation
- signal quality analytics
- candidate signal discovery

Primary file zones:

- `wallet_manager/strategies/alpha_engine.py`
- `wallet_manager/strategies/robust_alpha.py`
- `wallet_manager/signal_intelligence.py`
- `wallet_manager/signal_discovery.py`
- `wallet_manager/strategies/adaptive_ensemble.py`

### 3.4 Risk Layer

Responsibilities:

- position sizing
- capital throttling
- drawdown control
- regime-aware scaling
- kill-switches
- brake policy
- stress scenario logic

Primary file zones:

- `wallet_manager/strategies/deep_kelly_risk.py`
- `wallet_manager/strategies/bayesian_kelly.py`
- `wallet_manager/risk_intelligence.py`
- `wallet_manager/risk_committee.py`

### 3.5 Execution Layer

Responsibilities:

- route selection
- gas timing
- slippage estimation and calibration
- latency tracking
- execution quality measurement
- paper-vs-shadow-vs-live comparison

Primary file zones:

- `wallet_manager/strategies/smart_order_router.py`
- `wallet_manager/strategies/execution_quality.py`
- `wallet_manager/strategies/shadow_live.py`
- `wallet_manager/network/*`

### 3.6 Research Layer

Responsibilities:

- backtesting
- walk-forward validation
- regime attribution
- signal attribution
- model attribution
- threshold review

Primary file zones:

- `wallet_manager/strategies/wf_validator.py`
- `wallet_manager/research_brain.py`
- `scripts/backtest.py`

### 3.7 Observability Layer

Responsibilities:

- metrics API
- dashboard
- runtime summaries
- alerts
- anomaly detection
- audit trails

Primary file zones:

- `wallet_manager/metrics_server.py`
- `wallet_manager/alerting.py`
- `wallet_manager/reporting.py`

### 3.8 Promotion Layer

Responsibilities:

- experiment registry
- champion/challenger
- rollout gates
- promotion/demotion audit trail

Primary file zones:

- `wallet_manager/experiment_lab.py`
- `wallet_manager/strategies/model_registry.py`
- `wallet_manager/strategies/shadow_live.py`

## 4. Platform Contracts

Every mature phase should strengthen one or more platform contracts.

### 4.1 Runtime Contract

The same config plus same code should yield the same mode and startup behavior.

### 4.2 Data Contract

Every persisted table and dataset must have clear schema and intended use.

### 4.3 Research Contract

Every research result must be reproducible from explicit data, parameters, and
code version.

### 4.4 Signal Contract

Every signal must have:

- definition
- inputs
- output range
- confidence meaning
- quality measurement path

### 4.5 Risk Contract

Every position-sizing or veto mechanism must have:

- clear inputs
- clear outputs
- explicit rationale
- measurable impact

### 4.6 Execution Contract

Every execution path must produce enough telemetry to answer:

- what route was used
- what cost was estimated
- what cost was realized
- whether paper and shadow assumptions matched reality

### 4.7 Promotion Contract

Every promotion must be backed by:

- evidence
- threshold
- rollout stage
- rollback rule

## 5. Global Program Scorecard

The platform should be judged continuously against these scorecard dimensions.

### 5.1 Reliability

- startup success rate
- runtime error rate
- lint/type health
- test pass rate
- coverage

### 5.2 Sample Quality

- decision count
- trade count
- closed trade count
- skip reason concentration
- data freshness

### 5.3 Research Quality

- walk-forward stability
- reproducibility
- signal IC/IR
- regime breakdown availability
- execution-cost-adjusted metrics

### 5.4 Risk Quality

- drawdown responsiveness
- brake clarity
- stress scenario behavior
- capital protection consistency

### 5.5 Execution Quality

- slippage calibration
- route quality
- gas cost control
- shadow-live mismatch

### 5.6 Promotion Quality

- experiment throughput
- promotion discipline
- rollback quality
- challenger evidence quality

## 6. Phase Dependency Map

The correct dependency order is:

1. Core Reliability
2. Paper Trading Rehabilitation
3. Measurement and Memory
4. Data Layer
5. Research and Backtesting Engine
6. Alpha Factory
7. Risk Operating System
8. Execution Operating System
9. Experiment and Promotion Operating System
10. Continuous Learning

This order is mandatory because each phase depends on the previous one.

## 7. Phase 1 - Core Reliability

### 7.1 Goal

Make the production core reliable enough to support everything else.

### 7.2 Why It Matters

No advanced research platform survives if runtime, tests, typing, and startup
behavior are unstable.

### 7.3 Scope

- runtime breakages
- import/startup breakages
- type issues
- lint issues
- flaky tests
- coverage >= 85%

### 7.4 Primary File Zones

- `wallet_manager/orchestrator.py`
- `wallet_manager/task_factories.py`
- `wallet_manager/strategies/paper_engine.py`
- `wallet_manager/strategies/quantx_engine.py`
- `wallet_manager/security/telegram_commands.py`

### 7.5 Software Deliverables

- stable imports and startup path
- deterministic configuration loading
- testable helper surfaces
- stable CI-grade validation commands
- documented runtime boot path
- documented validation rhythm for local iteration versus checkpoint sweeps

### 7.6 Artifacts

- passing validation commands
- coverage report >= 85%
- stable startup doc path

### 7.7 KPIs

- `ruff check wallet_manager tests`
- `mypy wallet_manager --ignore-missing-imports`
- `pytest tests -q`
- `pytest tests --cov=wallet_manager --cov-report=term --cov-fail-under=85 -q`

Recommended validation cadence while Phase 1 is in progress:

- for a local code edit, first run targeted tests for the touched module or
  subsystem
- if a fix touches shared runtime surfaces, expand to the nearest relevant test
  cluster before rerunning the whole suite
- reserve the full four-command sweep for checkpoint validation, not every
  micro-edit

### 7.8 Exit Gate

Phase 1 is complete only if all four validation commands pass.

Those four commands must pass against the same code state.

If warnings remain but do not fail the commands, they must be documented in
repo docs so later phases do not silently inherit them.

### 7.9 Anti-Goals

Do not:

- add late-stage platform features while these checks are red
- hide failures under broad exception suppression
- accept weak coverage on core files

## 8. Phase 2 - Paper Trading Rehabilitation

### 8.1 Goal

Make `robust_paper` produce enough high-quality samples to support research and
promotion decisions.

### 8.2 Why It Matters

The current system is over-filtered and sample-starved. A platform cannot learn
from almost no trades.

### 8.3 Known Current Symptoms

- `RISK_BLOCKED(normal)`
- `RISK_COMMITTEE:ModelDisagreement`
- `TRADE_BUDGET_TOO_SMALL`
- occasional stress-test rejection

### 8.4 Scope

- simplify veto chains in paper mode
- separate paper-mode sizing from live-mode sizing
- reduce duplicate filters
- make skip reasons analytically trustworthy
- preserve safety while restoring sample flow

### 8.5 Primary File Zones

- `wallet_manager/strategies/paper_engine.py`
- `wallet_manager/strategies/deep_kelly_risk.py`
- `wallet_manager/risk_intelligence.py`
- `wallet_manager/strategies/bayesian_kelly.py`
- `wallet_manager/task_factories.py`

### 8.6 Software Deliverables

- explicit paper-mode sizing policy
- reduced veto duplication
- structured skip reasoning
- runtime-visible sample starvation metrics
- documented baseline-vs-post-change comparison method

### 8.7 Artifacts

- cleaner reason distributions
- healthier paper decision throughput
- trustworthy decision analytics
- a repo-documented before/after measurement snapshot for the chosen paper window

### 8.8 KPIs

Judge these against an explicit before/after comparison window of the same mode,
similar runtime duration, and documented config assumptions.

- decision volume rises versus the recorded baseline window
- closed-trade count rises enough to support ongoing analysis, not just raw
  decision spam
- no single avoidable sizing or duplicate-filter failure dominates the skip
  distribution for the representative window
- `TRADE_BUDGET_TOO_SMALL` and duplicate veto patterns fall versus baseline
  without disabling all protection
- skip reasons become more analytically trustworthy, not noisier

### 8.9 Exit Gate

Phase 2 is complete only when a repo-documented baseline-vs-post-change
comparison shows that paper mode is no longer dominated by avoidable
sizing/filter deadlocks, preserves explicit safety controls, and generates
enough decisions and closed trades for serious analysis.

Do not declare Phase 2 complete from terminal impressions alone.

### 8.10 Anti-Goals

Do not:

- brute-force trading by disabling all risk
- treat paper mode as live mode
- confuse more trades with better trades

## 9. Phase 3 - Measurement and Memory

### 9.1 Goal

Know exactly what the system decided and why.

### 9.2 Scope

- decision journal integrity
- no-trade persistence
- regime/setup tagging
- audit snapshots
- runtime and research summaries

### 9.3 Primary File Zones

- `wallet_manager/metrics_server.py`
- `wallet_manager/strategies/paper_engine.py`
- `wallet_manager/resilience/analytics.py`

### 9.4 Software Deliverables

- stable decision/event schemas
- queryable runtime summaries
- persistent audit snapshots

### 9.5 Artifacts

- durable DB tables
- stable research API outputs
- dashboard-backed decision context

### 9.6 KPIs

- every trade and skip is explainable after the fact
- API/dashboard values match persisted data

### 9.7 Exit Gate

Operators can answer "what happened and why" without relying on terminal logs.

## 10. Phase 4 - Data Layer

### 10.1 Goal

Upgrade from ad hoc runtime data to quant-grade data infrastructure.

### 10.2 Scope

- versioned historical datasets
- replayable data windows
- feature persistence
- tick storage
- gas and liquidity snapshots

### 10.3 Primary File Zones

- `wallet_manager/tick_collector.py`
- `wallet_manager/resilience/analytics.py`
- `scripts/data/*`
- data databases

### 10.4 Software Deliverables

- explicit data contracts
- reproducible dataset versions
- feature-store-like persistence

### 10.5 Artifacts

- named dataset versions or slice IDs
- replayable research inputs
- stable feature generation path

### 10.6 KPIs

- same dataset version reproduces same research result
- feature generation does not drift silently

### 10.7 Exit Gate

Research can be rerun deterministically from persisted data artifacts.

## 11. Phase 5 - Research and Backtesting Engine

### 11.1 Goal

Move from anecdotal paper trading to evidence-backed research.

### 11.2 Scope

- event-driven backtests
- slippage-aware backtests
- latency-aware backtests
- gas-aware backtests
- liquidity-aware backtests
- walk-forward validation

### 11.3 Primary File Zones

- `wallet_manager/strategies/wf_validator.py`
- `wallet_manager/research_brain.py`
- `scripts/backtest.py`

### 11.4 Software Deliverables

- event-driven research engine
- reproducible walk-forward orchestration
- standardized research outputs

### 11.5 Artifacts

- backtest reports
- walk-forward reports
- comparable model/strategy evaluations

### 11.6 KPIs

- realistic cost-adjusted strategy comparison
- reproducible research output packages

### 11.7 Exit Gate

Every promoted behavior has reproducible research evidence behind it.

## 12. Phase 6 - Alpha Factory

### 12.1 Goal

Systematically create, score, compare, and retire alpha sources.

### 12.2 Scope

- feature registry
- signal registry
- IC and IR tracking
- signal decay analysis
- signal redundancy analysis
- model contribution tracking
- champion/challenger logic

### 12.3 Primary File Zones

- `wallet_manager/signal_intelligence.py`
- `wallet_manager/signal_discovery.py`
- `wallet_manager/strategies/model_registry.py`
- `wallet_manager/continuous_learning.py`

### 12.4 Software Deliverables

- explicit signal metadata
- signal quality scorecards
- model contribution scorecards
- challenger comparison logic

### 12.5 Artifacts

- per-signal IC/IR outputs
- signal decay reports
- model contribution reports
- challenger evidence logs

### 12.6 KPIs

- weak signals are discoverable quickly
- confidence quality is measurable, not assumed
- underperforming models can be retired with evidence

### 12.7 Exit Gate

The platform can clearly say which signals and models are adding value.

## 13. Phase 7 - Risk Operating System

### 13.1 Goal

Make risk a first-class operating layer above alpha.

### 13.2 Scope

- position-sizing policy
- exposure limits
- drawdown acceleration
- scenario stress behavior
- kill-switch policy
- regime-aware protection

### 13.3 Primary File Zones

- `wallet_manager/risk_intelligence.py`
- `wallet_manager/risk_committee.py`
- `wallet_manager/strategies/deep_kelly_risk.py`

### 13.4 Software Deliverables

- centralized brake policy
- explicit risk state outputs
- explainable veto narratives

### 13.5 Artifacts

- stress reports
- drawdown state reports
- risk state API payloads

### 13.6 KPIs

- risk reductions happen before damage compounds
- veto logic is measurable and auditable

### 13.7 Exit Gate

Risk can be reasoned about as a system, not as scattered conditionals.

## 14. Phase 8 - Execution Operating System

### 14.1 Goal

Optimize execution independently from alpha generation.

### 14.2 Scope

- route quality
- slippage calibration
- gas timing
- latency measurement
- execution cost attribution
- shadow/live comparison

### 14.3 Primary File Zones

- `wallet_manager/strategies/smart_order_router.py`
- `wallet_manager/strategies/execution_quality.py`
- `wallet_manager/strategies/shadow_live.py`
- `wallet_manager/network/*`

### 14.4 Software Deliverables

- execution telemetry contracts
- route quality measurement
- cost attribution per trade

### 14.5 Artifacts

- TCA-style trade records
- slippage calibration outputs
- shadow-vs-paper comparison reports

### 14.6 KPIs

- realized execution quality is measurable
- paper assumptions can be compared against shadow behavior

### 14.7 Exit Gate

Execution quality becomes its own measurable optimization surface.

## 15. Phase 9 - Experiment and Promotion Operating System

### 15.1 Goal

Promote only evidence-backed behavior through a controlled rollout path.

### 15.2 Scope

- experiment registry
- hypothesis registry
- rollout stages
- promotion thresholds
- rollback policy

### 15.3 Primary File Zones

- `wallet_manager/experiment_lab.py`
- `wallet_manager/strategies/model_registry.py`
- `wallet_manager/strategies/shadow_live.py`

### 15.4 Software Deliverables

- stage-gated rollout logic
- promotion and demotion rules
- challenger observation pipeline

### 15.5 Artifacts

- explicit rollout records
- promotion decisions with evidence
- rollback audit trail

### 15.6 KPIs

- no uncontrolled strategy promotion
- promotion decisions are reproducible and reviewable

### 15.7 Exit Gate

The platform can safely move behavior from paper to capital with evidence.

## 16. Phase 10 - Continuous Learning

### 16.1 Goal

Keep the system adaptive without letting it drift into chaos.

### 16.2 Scope

- drift review
- setup decay tracking
- threshold retuning review
- weekly change archive
- feature importance drift

### 16.3 Primary File Zones

- `wallet_manager/continuous_learning.py`
- `wallet_manager/research_brain.py`
- `wallet_manager/reporting.py`

### 16.4 Software Deliverables

- periodic evidence review loop
- controlled candidate retuning flow
- historical change memory

### 16.5 Artifacts

- weekly reports
- drift reports
- threshold review summaries
- feature decay alerts

### 16.6 KPIs

- meaningful drift is visible
- tuning suggestions are evidence-backed
- changes are archived and reviewable

### 16.7 Exit Gate

The platform evolves with evidence instead of intuition.

## 17. 90-Day Execution Plan

This section translates the roadmap into an execution cadence.

The calendar does not override the phase gates.

If a phase slips, the correct action is to finish the blocking phase rather than
pretend the later calendar milestone has priority.

### 17.1 Days 0-30

Mandatory goals:

- finish Phase 1
- finish Phase 2
- stabilize Phase 3 foundations

Expected outcomes:

- reliable core
- healthy paper-mode sample flow
- trustworthy runtime and decision analytics

### 17.2 Days 31-60

Mandatory goals:

- finish Phase 4
- materially advance Phase 5
- begin Phase 6 instrumentation

Expected outcomes:

- replayable datasets
- reproducible backtesting
- initial signal quality visibility

### 17.3 Days 61-90

Mandatory goals:

- complete Phase 5
- make Phase 6 and Phase 7 operational
- begin Phase 8 and Phase 9 foundations

Expected outcomes:

- evidence-backed research loop
- measurable risk layer
- measurable execution layer
- controlled rollout path toward shadow and tiny live

## 18. Definition of Strategic Success

The roadmap is succeeding if, over time, the repository becomes a platform
where:

- runtime failures are rare and explainable
- paper mode produces enough quality samples
- research is reproducible
- signal quality is measurable
- risk behavior is explainable
- execution quality is measurable
- promotions are disciplined
- operators and AI agents can continue work without inventing new structure

## 19. Permanent Anti-Patterns

These are permanent "do not do this" rules.

Do not:

- skip ahead to later phases because they sound more advanced
- add more models before measuring current signal quality
- confuse more trades with better research
- confuse larger confidence values with better calibration
- bury risk behavior inside strategy-specific ad hoc branches
- bury operational truth in chat instead of repo docs
- move behavior toward live capital without evidence and rollback logic

## 20. Current Priority

At the current state of the repository, the only valid top priorities are:

1. Phase 1 - Core Reliability
2. Phase 2 - Paper Trading Rehabilitation

Everything else is secondary until these are finished.

Roadmap edits are allowed only if they make these priorities easier to execute
without weakening the current gate.
