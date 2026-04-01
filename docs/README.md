# Documentation Hub

This directory is organized for **spec-driven development (SDD)** and **LLM-assisted implementation**.

## Document Map

### Product-oriented docs
- [product/product-context.md](product/product-context.md): Product scope, personas, goals, and non-goals.
- [product/spec-driven-workflow.md](product/spec-driven-workflow.md): How to go from idea → approved spec → code.
- [product/decision-log.md](product/decision-log.md): Decision choices with alternatives and trade-offs.

### Technical docs
- [technical/technical-context.md](technical/technical-context.md): LLM-oriented technical context for the repository.
- [technical/architecture.md](technical/architecture.md): Port/adapters architecture, data flow, and Mermaid diagrams.
- [technical/extension-playbook.md](technical/extension-playbook.md): How to add new adapters/ports safely.

### Spec templates
- [specs/feature-spec-template.md](specs/feature-spec-template.md): Feature spec template.
- [specs/adapter-spec-template.md](specs/adapter-spec-template.md): Adapter implementation spec template.

### Implemented specs
- [specs/benchmark-dataset-metrics-option-c.md](specs/benchmark-dataset-metrics-option-c.md): Benchmark dataset coverage, metric implementations, and first-N local validation.

## How to use this as the source of truth
1. Start from product intent in `product/`.
2. Write or update a spec in `docs/specs/`.
3. Validate against architecture constraints in `technical/`.
4. Implement only after acceptance criteria and test strategy are explicit.
