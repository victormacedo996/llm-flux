# Decision Log

This file captures product/engineering decision choices in a compact ADR-style format.

## Status legend
- `proposed`
- `accepted`
- `superseded`
- `rejected`

---

## DEC-001 — Port/Adapters as primary extension strategy
- Status: `accepted`
- Date: 2026-03-31
- Context: The framework needs fast integration of new compression/healing/profiling methods.
- Decision: Keep core interfaces stable and add behavior through adapters implementing ports.
- Alternatives considered:
  - hard-coded strategy switch statements
  - plugin registry with dynamic imports only
- Why chosen:
  - lower coupling
  - easier testability
  - supports Open/Closed principle
- Consequences:
  - requires clear contracts and adapter validation
  - small upfront discipline, large long-term speed gain

## DEC-002 — Declarative linear pipeline as default
- Status: `accepted`
- Date: 2026-03-31
- Context: Most experiments are sequential and easier to debug linearly.
- Decision: Keep `PipelineStep` as linear order converted to DAG internally.
- Alternatives considered:
  - full graph authoring from the beginning
- Why chosen:
  - easier onboarding
  - readable examples
- Consequences:
  - branching/fan-out is future work

## DEC-003 — Structured result object as experiment source of truth
- Status: `accepted`
- Date: 2026-03-31
- Context: Experiments need comparable outputs across runs.
- Decision: Keep `PipelineRunResult` + `ProfilingResult` as canonical output models.
- Alternatives considered:
  - ad-hoc logs only
  - external experiment DB first
- Why chosen:
  - reproducibility
  - simple markdown/json/html generation
- Consequences:
  - schema evolution must stay backward compatible

## DEC-004 — English-only docs for LLM consistency
- Status: `accepted`
- Date: 2026-03-31
- Context: AI-assisted development quality is sensitive to language fragmentation.
- Decision: Keep all specs and architecture docs in English.
- Alternatives considered:
  - bilingual docs
- Why chosen:
  - consistent prompt context for coding assistants
  - less duplication
- Consequences:
  - contributors should keep terminology in English

## DEC-005 — Product and technical docs split
- Status: `accepted`
- Date: 2026-03-31
- Context: LLMs and humans need clear separation between intent and implementation.
- Decision: Maintain two documentation layers:
  - `docs/product/` for goals, process, decisions
  - `docs/technical/` for architecture/contracts/extension details
- Alternatives considered:
  - single monolithic context file
- Why chosen:
  - easier retrieval for targeted prompts
  - lower cognitive load
- Consequences:
  - cross-links must be maintained

## DEC-006 — Modular report templates in nested `report/` tree
- Status: `accepted`
- Date: 2026-03-31
- Context: The HTML reporter template had grown into a single large Jinja file, making section-level edits harder and increasing context pressure for AI-assisted maintenance.
- Decision: Move the HTML report entry template into `llm_flux/core/templates/report/` and compose it from focused partials/macros while preserving rendered visual and DOM equivalence.
- Alternatives considered:
  - keep a single monolithic template and only add comments
  - move CSS/JS into external static assets
  - reshape the Python render context during the refactor
- Why chosen:
  - lowers friction for adding/refactoring sections
  - keeps the report self-contained
  - limits the change to the templating layer
- Consequences:
  - include ordering and DOM identifiers become compatibility constraints
  - maintainer documentation must be kept in sync with the template tree

## DEC-007 — Option C benchmark expansion with deterministic first-N sampling
- Status: `accepted`
- Date: 2026-03-31
- Context: The thesis workflow requires concrete benchmark outputs quickly, including QA-oriented metrics, while still allowing CPU-only local validation.
- Decision: Expand benchmark coverage to include perplexity + QA first (Option C), implement requested metric families, and standardize dataset slicing as deterministic first-N rows.
- Alternatives considered:
  - random sampled N rows
  - perplexity-only benchmarking in first iteration
  - introducing a separate benchmark orchestration layer
- Why chosen:
  - matches advisor-driven need for early measurable results
  - deterministic slices improve reproducibility in local validation
  - reuses existing profiling and result models with minimal structural churn
- Consequences:
  - first-N may be less representative than randomized subsets
  - some dataset evaluators require schema fallbacks to remain robust
