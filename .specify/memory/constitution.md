<!--
Sync Impact Report
==================
Version change: 0.0.0 → 1.0.0
Bump rationale: MAJOR — initial ratification of project constitution

Added principles:
  - I. Deterministic Simulation Integrity (new)
  - II. Conservation & Correctness (new)
  - III. Behavioral Test-Driven Development (new)
  - IV. Library-CLI Separation (new)
  - V. Single-Method Interface Design (new)
  - VI. Module-Scoped Configuration (new)
  - VII. Documentation Single Source of Truth (new)
  - VIII. Defensive Validation at Every Boundary (new)

Added sections:
  - System Invariants (9 invariants from invariants.md)
  - Development Workflow & Quality Gates (PR workflow, convergence)

Templates requiring updates:
  - .specify/templates/plan-template.md — ✅ Constitution Check section
    already references constitution gates generically; no update needed
  - .specify/templates/spec-template.md — ✅ no constitution references;
    no update needed
  - .specify/templates/tasks-template.md — ✅ no constitution references;
    no update needed

Follow-up TODOs: none
-->

# BLIS Constitution

## Core Principles

### I. Deterministic Simulation Integrity

The simulator MUST produce byte-identical stdout for the same seed
and configuration across runs. Wall-clock timing MUST go to stderr.

Non-negotiable rules:
- Map iteration feeding float sums or output ordering MUST sort keys
  first (R2, INV-6).
- All randomness MUST flow through `PartitionedRNG` — never
  wall-clock-dependent sources.
- Stateful scorers MUST NOT introduce non-deterministic internal
  state.
- Floating-point accumulation order MUST be fixed (sorted keys or
  deterministic iteration).

**Rationale:** Determinism (INV-6) is the foundation of capacity
planning and reproducible experiments. A non-deterministic simulator
cannot be trusted for policy comparison.

### II. Conservation & Correctness

Every resource tracked by the simulator MUST satisfy a conservation
law verifiable at simulation end.

Non-negotiable rules:
- Request conservation (INV-1):
  `injected == completed + queued + running + dropped_unservable`.
  Full pipeline: `num_requests == injected + rejected`.
- KV cache conservation (INV-4):
  `allocated_blocks + free_blocks == total_blocks` at all times.
- Causality (INV-5):
  `arrival <= enqueue <= schedule <= completion` per request.
- Every error path MUST return error, panic, or increment a counter
  — never silently drop data (R1).
- Resource-allocating loops MUST rollback on mid-loop failure (R5).

**Rationale:** Conservation invariants are the simulator's
correctness contract. Issue #183 demonstrated that a single silently
dropped request went undetected for months because golden tests
encoded the wrong value as expected.

### III. Behavioral Test-Driven Development

All features MUST be developed test-first with behavioral contracts.

Non-negotiable rules:
- Write GIVEN/WHEN/THEN behavioral contracts before implementation.
- Tests MUST assert observable behavior, not internal structure.
  Prohibited: type assertions (`policy.(*ConcreteType)`), internal
  field access, exact formula reproduction. Required: observable
  output, invariant verification, ordering/ranking assertions.
- Every golden test MUST have a companion invariant test verifying
  a system law (R7). Golden tests answer "did the output change?"
  but not "is the output correct?"
- Refactor survival test: "Would this test still pass if the
  implementation were completely rewritten but the behavior
  preserved?" If no, the test is structural — rewrite it.
- Individual test budget: no single test exceeding 5 seconds without
  `testing.Short()`. Total `go test ./...` target: under 60 seconds.

**Rationale:** BDD/TDD prevents the class of bug where golden tests
perpetuate existing defects (issue #183). Behavioral tests survive
refactoring and verify what the system guarantees, not how it works.

### IV. Library-CLI Separation

`sim/` is a library — it MUST never terminate the process. Only
`cmd/` may terminate.

Non-negotiable rules:
- `sim/` MUST NOT call `os.Exit`, `logrus.Fatalf`, or any
  process-terminating function (R6).
- Error handling boundaries: CLI (`cmd/`) uses `logrus.Fatalf` for
  user errors. Library (`sim/`) uses `panic()` for invariant
  violations and `error` return for recoverable failures.
- Cluster-level policies (admission, routing) receive `*RouterState`
  with global view. Instance-level policies (priority, scheduler)
  receive only local data. Never leak cluster state to instance-level
  code.
- Dependency direction: `cmd/ → sim/cluster/ → sim/`. `sim/` MUST
  NOT import subpackages.
- stdout carries deterministic results; stderr carries diagnostics.

**Rationale:** Library-CLI separation enables embedding, testing,
and adapter use cases. Terminating from library code prevents test
isolation and forces callers into a single error-handling strategy.

### V. Single-Method Interface Design

Interfaces MUST be defined by behavioral contract, not by one
implementation's data model.

Non-negotiable rules:
- Single-method interfaces where possible (`AdmissionPolicy`,
  `RoutingPolicy`, `PriorityPolicy`, `InstanceScheduler`) (R13).
- New interfaces MUST accommodate at least two implementations, even
  if only one exists today. No methods that only make sense for one
  backend.
- Query methods MUST be pure — no side effects, no state mutation.
  Separate `Get()` and `Consume()` for query-and-clear.
- No method MUST span multiple module responsibilities (R14).
  Extract each concern into its module's interface.
- Factory functions MUST validate inputs: `IsValid*()` check +
  switch/case + panic on unknown.

**Rationale:** Single-method, behavioral interfaces minimize coupling
and maximize composability. R13 was born from the KVStore interface
encoding vLLM's block-level model, blocking alternative backends.

### VI. Module-Scoped Configuration

Configuration MUST be grouped by module, not added to monolithic
structs.

Non-negotiable rules:
- `SimConfig` is composed of 6 embedded sub-configs:
  `KVCacheConfig`, `BatchConfig`, `LatencyCoeffs`,
  `ModelHardwareConfig`, `PolicyConfig`, `WorkloadConfig` (R16).
- Factory signatures MUST accept the narrowest sub-config (e.g.,
  `NewKVStore(KVCacheConfig)`, not `NewKVStore(SimConfig)`).
- Each module's config MUST be independently specifiable and
  validatable.
- Every struct constructed in multiple places needs a canonical
  constructor. Struct literals appear in exactly one place (R4).
  Before adding a field, grep for ALL construction sites.

**Rationale:** Module-scoped configuration resolved the 23-field
monolithic `SimConfig` that mixed hardware, model, simulation, and
policy concerns (issue #350). Narrow factory signatures make
dependencies explicit and testing easier.

### VII. Documentation Single Source of Truth

Every piece of documentation MUST live in exactly one canonical
location.

Non-negotiable rules:
- Other files MAY contain working copies (summaries) with explicit
  canonical-source headers:
  `> Canonical source: [path]. If this section diverges, [path] is
  authoritative.`
- When updating any standard, invariant, rule, or recipe: update
  the canonical source FIRST, then update working copies.
- The source-of-truth map in `docs/contributing/standards/principles.md`
  MUST be kept current. It lists every canonical source and all known
  working copies.

**Rationale:** Multiple copies of the same information drift over
time. The canonical-source pattern ensures readers always know which
version to trust, even when working copies are stale.

### VIII. Defensive Validation at Every Boundary

Every numeric parameter MUST be validated — at CLI flags AND library
constructors.

Non-negotiable rules:
- Validate for: zero, negative, NaN, Inf, and empty string (R3).
  CLI: `logrus.Fatalf` in `cmd/root.go`. Library: `panic` or `error`
  in the constructor. Validation MUST appear before the first
  consumption site.
- Division where the denominator derives from runtime state MUST
  guard against zero (R11).
- YAML config structs MUST use `*float64` (pointer) for fields where
  zero is a valid user value (R9).
- YAML parsing MUST use `yaml.KnownFields(true)` — typos MUST
  cause errors (R10).
- Pre-check estimates MUST be consistent with (at least as permissive
  as) the actual operation they guard (R22).
- CLI flag values MUST NOT be silently overwritten by defaults.yaml
  — always check `cmd.Flags().Changed()` before applying a default
  (R18).

**Rationale:** Missing validation caused infinite loops (Rate=0),
wrong results (NaN weights), and silent misconfiguration. Library
callers bypass CLI validation entirely, so constructors MUST
independently validate.

## System Invariants

Nine invariants that MUST hold at all times during and after
simulation. Canonical source:
`docs/contributing/standards/invariants.md`.

| ID | Name | Statement |
|----|------|-----------|
| INV-1 | Request Conservation | `injected == completed + queued + running + dropped` |
| INV-2 | Request Lifecycle | Transitions: queued → running → completed only |
| INV-3 | Clock Monotonicity | Simulation clock never decreases |
| INV-4 | KV Cache Conservation | `allocated + free == total` at all times |
| INV-5 | Causality | `arrival <= enqueue <= schedule <= completion` |
| INV-6 | Determinism | Same seed → byte-identical stdout |
| INV-7 | Signal Freshness | Tiered freshness hierarchy for routing signals |
| INV-8 | Work-Conserving | No idle while WaitQ non-empty |
| INV-9 | Oracle Knowledge Boundary | Control plane MUST NOT read `OutputTokens` |

Invariant violations are correctness bugs — they MUST be fixed
before any PR merges.

## Development Workflow & Quality Gates

Canonical source: `docs/contributing/pr-workflow.md`.

Every PR follows this pipeline:

1. **Worktree isolation** — create worktree BEFORE any work.
2. **Micro-plan** — behavioral contracts + TDD task breakdown.
3. **Plan review** — 10-perspective convergence protocol (zero
   CRITICAL + zero IMPORTANT = converged).
4. **Human approval** — plan approved before implementation begins.
5. **TDD implementation** — test → fail → implement → pass → lint
   → commit per task.
6. **Code review** — 10-perspective convergence protocol.
7. **Self-audit** — 10-dimension critical thinking (no agent).
8. **Commit + PR** — with behavioral contract references.

### Antipattern Rules

23 rules (R1–R23) enforced at PR template, micro-plan Phase 8,
and pre-commit self-audit. Each traces to a real bug. Canonical
source: `docs/contributing/standards/rules.md`.

Priority tiers:
- **Critical** (R1, R4, R5, R6, R11, R19, R21): correctness —
  violations produce silent data loss, panics, or infinite loops.
- **Important** (R2, R3, R7–R10, R13, R14, R17, R18, R20, R22,
  R23): quality — violations produce non-determinism, validation
  gaps, or interface debt.
- **Hygiene** (R12, R15, R16): maintenance — violations produce
  stale references or config sprawl.

## Governance

This constitution supersedes ad-hoc practices. All PRs and reviews
MUST verify compliance with the Core Principles and System
Invariants.

### Amendment Procedure

1. Propose amendment via GitHub issue with rationale and evidence.
2. Update canonical source (`docs/contributing/standards/`) first.
3. Update this constitution to reflect the change.
4. Increment version per semantic versioning:
   - MAJOR: principle removal or backward-incompatible redefinition.
   - MINOR: new principle added or materially expanded guidance.
   - PATCH: clarifications, wording fixes, non-semantic refinements.
5. Update `LAST_AMENDED_DATE`.

### Compliance

- Every PR review MUST check Critical antipattern rules (R1, R4,
  R5, R6, R11, R19, R21) at minimum.
- Every convergence review round MUST verify invariant compliance.
- Complexity beyond what principles permit MUST be justified in a
  Complexity Tracking table (see plan template).
- Runtime development guidance: `CLAUDE.md` (working copy of
  principles, rules, and invariants with canonical-source headers).

**Version**: 1.0.0 | **Ratified**: 2026-03-12 | **Last Amended**: 2026-03-12
