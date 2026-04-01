# Feature Spec

## Metadata
- Spec ID: `SPEC-0007`
- Title: Benchmark dataset coverage + metric implementations (Option C)
- Author: GitHub Copilot
- Date: 2026-03-31
- Status: `implemented`
- Related decisions: [docs/product/decision-log.md](../product/decision-log.md)

## 1. Problem statement
The framework had only one concrete benchmark (`lambada` perplexity), which was insufficient to produce thesis-ready comparative results across compression stages. A broader benchmark surface was required, including QA and reasoning tasks, with a local CPU-friendly validation mode.

## 2. Goals
- Implement concrete metric evaluators requested in the dataset/metric table.
- Add concrete dataset usage methods for all listed benchmark datasets.
- Support deterministic first-N slicing for local end-to-end validation.
- Keep architecture aligned with existing ports/adapters and profiling outputs.

## 3. Non-goals
- Building a distributed benchmark runner.
- Adding external benchmark harness dependencies.
- Guaranteeing leaderboard-equivalent formatting for every dataset variant.

## 4. Scope
### In scope
- `ModelPerformanceBenchmarker` test methods and metric logic.
- First-N semantics for `DatasetConfig.max_samples` in HF and local adapters.
- Profiling adapter mapping for non-perplexity benchmark scores.
- Example script for Option C local validation.
- Documentation updates.

### Out of scope
- Full benchmark prompt-engineering optimization.
- Parallel benchmark execution orchestration.

## 5. User scenarios
- As a researcher, I run a compression pipeline and compare quality impacts across perplexity and QA metrics.
- As a local developer without GPU, I validate the full process with first N rows.

## 6. Proposed solution
- Extend benchmark test registry to include all requested datasets.
- Implement metric handlers: Perplexity, Accuracy, Number Extraction Accuracy, F1, MC2, Pass@1.
- Add `metric_name` to structured accuracy benchmark result.
- Keep `limit_test_samples` as first-N control and route it through benchmark execution.

## 7. Decision choices
- Choice: first-N deterministic slicing.
  - Alternatives: random sampling with seed.
  - Selected option: first-N.
  - Rationale: explicit user request and stable local reproducibility.
  - Trade-offs: may be less representative than randomized samples.
- Choice: Option C prioritization (Perplexity + QA first).
  - Alternatives: full heavy benchmark focus first.
  - Selected option: Option C.
  - Rationale: fast path to visible results in advisor-guided iteration.
  - Trade-offs: some tests (e.g., code tasks) remain lightweight approximations.

## 8. Technical impact
- Modules affected:
  - `llm_flux/profiling/model_benchmarker.py`
  - `llm_flux/profiling/types/benchmark.py`
  - `llm_flux/profiling/adapters/comprehensive.py`
  - `llm_flux/datasets/huggingface.py`
  - `llm_flux/datasets/local.py`
  - `llm_flux/datasets/port.py`
  - `examples/e2e_option_c_validation.py`
- New dependencies: none.
- Data model changes: `AccuracyTestResult.metric_name`.
- Backward compatibility notes: existing `accuracy` field preserved.

## 9. Acceptance criteria (testable)
- [x] Benchmark registry includes all CSV datasets.
- [x] Metrics implemented: accuracy, F1, MC2, Pass@1, perplexity, number extraction accuracy.
- [x] First-N row slicing works through benchmark sample limits.
- [x] Example script demonstrates Option C local end-to-end flow.

## 10. Validation plan
- Run a local Option C benchmark example with tiny model.
- Verify benchmark JSON includes `metric_name` and per-test entries.
- Verify adapter sampling behavior returns first N rows.

## 11. Rollout plan
- Merge feature.
- Run Option C script to produce initial thesis baseline artifacts.
- Extend prompts/scoring with dataset-specific refinements as needed.

## 12. Risks and mitigations
- Risk: dataset schema variations.
  - Mitigation: robust field fallbacks and split fallbacks.
- Risk: code benchmark execution instability.
  - Mitigation: catch exceptions and score only runnable items.

## 13. Open questions
- Should pass@1 execution be sandboxed in a subprocess with timeout in the next iteration?
