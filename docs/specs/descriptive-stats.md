# Feature Spec — Descriptive Statistics for All Metrics

## Metadata
- Spec ID: `SPEC-0003`
- Title: Descriptive Statistics for Computational and Model-Performance Metrics
- Author: Copilot (spec-driven session)
- Date: 2026-03-31
- Status: `implemented`
- Related decisions: `docs/product/decision-log.md`

---

## 1. Problem statement

The framework currently emits only scalar summaries for every measurement:
- **Inference timing** — mean, min, max; p50/p95/p99 are *approximated* as mean/max/max (incorrect).
- **Perplexity** — only `mean_perplexity`; the raw per-sample list is stored but never summarised.
- **Accuracy benchmarks** (squad F1, ARC, GSM8K, TruthfulQA MC2, …) — per-example scores are *discarded* immediately after the aggregate is computed.

A master's thesis paper requires distributional evidence: confidence intervals, spread, and tail behaviour for all measured quantities. Scalar summaries alone are insufficient to support statistical claims.

---

## 2. Goals

1. Add `DescriptiveStats` (mean, std, min, p5, p25, median/p50, p75, p95, p99, max, n) to every metric that is sampled more than once.
2. Preserve per-example scores for all accuracy/perplexity benchmarks in the serialised JSON.
3. Expose all statistics in the interactive HTML report — stat cards, extended table columns, and chart confidence bands.
4. Fix the three incorrect latency approximations in `ComprehensiveProfilingAdapter`.

---

## 3. Non-goals

- Cross-run aggregation (statistics across different compression pipeline stages are not merged).
- Statistical significance testing (t-tests, effect sizes) — deferred to a separate feature.
- Changing benchmark dataset splits or evaluation protocols.

---

## 4. Scope

### In scope
- `DescriptiveStats` shared Pydantic model (`llm_flux/profiling/types/stats.py`)
- `InferencePerformanceInfo` — add p5/p50/p95/p99/std/raw_times
- `ComputePerplexityForDatasetReturn` — add `stats: DescriptiveStats`
- `AccuracyTestResult` — add `per_example_scores: List[float]` and `stats: DescriptiveStats | None`
- All 13 benchmark methods in `ModelPerformanceBenchmarker` — preserve per-example scores
- `LatencyMetrics` — add min_ms, std_ms, p5_ms, max_ms
- `ComprehensiveProfilingAdapter` — fix p50/p95/p99 mappings
- HTML reporter Python logic — forward all new fields
- 7 HTML template partials — display statistics

### Out of scope
- Hardware profiling statistics (single measurement per run by design)
- LLM structural profile statistics

---

## 5. User scenarios

- **Scenario A — Dissertation table**: researcher exports `results.json` and reads `latency.stats.p95` and `accuracy.stats.std` to report 95th-percentile latency and standard deviation of accuracy.
- **Scenario B — HTML review**: researcher opens `report.html`, expands the "Inference Benchmarks" accordion, and sees a full stat-card grid: Mean, Std, P5, P50, P95, P99, Min, Max.
- **Scenario C — Perplexity distribution**: per-sample perplexity values are shown as chips alongside their distributional summary (min, max, std).
- **Scenario D — Per-example accuracy analysis**: per-example F1/accuracy/MC2 scores are preserved in JSON; researchers can reconstruct the distribution offline.

---

## 6. Proposed solution

### 6.1 Shared `DescriptiveStats` model

```python
# llm_flux/profiling/types/stats.py
class DescriptiveStats(BaseModel):
    n: int
    mean: float; std: float; min: float
    p5: float; p25: float; median: float; p75: float; p95: float; p99: float
    max: float

    @classmethod
    def from_values(cls, values: List[float]) -> "DescriptiveStats": ...
```

Uses `numpy` (already a dependency). Sample std (`ddof=1`) for n > 1; zeros for empty input.

### 6.2 Data flow

```
InferencePerformanceBenchmarker.time_inference()
  times: list[float]  ← raw wall-clock samples
  → InferencePerformanceInfo (avg, std, min, p5, p50, p95, p99, max, raw_times)

ModelPerformanceBenchmarker._squad_f1() / _multiple_choice_accuracy() / ...
  per_example_scores: list[float]  ← preserved before discard
  → AccuracyTestResult (accuracy, per_example_scores, stats)

ModelPerformanceBenchmarker.evaluate_perplexity()
  all_perplexities: list[float]  ← already kept
  → ComputePerplexityForDatasetReturn (mean, stats)

ComprehensiveProfilingAdapter.profile()
  InferencePerformanceInfo → LatencyMetrics (correct p50/p95/p99)

PipelineRunResult.save_json()  ← no change needed; Pydantic serialises all fields

html_reporter._extract_inference_info()  ← forward all new fields
html_reporter._extract_benchmarks_info() ← forward stats + per_example_scores
html_reporter._render_template()         ← pass new JS arrays
```

---

## 7. Decision choices

| Choice | Alternatives | Selected | Rationale |
|--------|-------------|----------|-----------|
| Shared `DescriptiveStats` model vs. duplicate fields per type | Duplicate fields on each type | Shared model | Single source of truth; consistent field names across JSON |
| Sample std (ddof=1) vs population std | Population std | Sample std | n is always small (≤ N benchmark examples); sample estimate is more conservative |
| Recompute stats in reporter vs. forward pre-computed | Recompute | Forward pre-computed | Avoids numpy in the templating layer; stats are always authoritative from collection |
| Store `raw_times` in `InferencePerformanceInfo` | Discard after stats | Store | Enables offline reanalysis; adds ~40 bytes per float |

---

## 8. Technical impact

- **Modules affected**: `profiling/types/stats.py` (new), `profiling/types/benchmark.py`, `profiling/types/inference.py`, `profiling/types/__init__.py`, `profiling/inference_benchmarker.py`, `profiling/model_benchmarker.py`, `core/profiling.py`, `profiling/adapters/comprehensive.py`, `core/html_reporter.py`, 7 Jinja2 templates.
- **New dependencies**: none (numpy already required).
- **Data model changes**: `LatencyMetrics` gains 4 new optional fields; `AccuracyTestResult` gains 2 new optional fields; `InferencePerformanceInfo` gains 7 new fields.
- **Backward compatibility**: all new fields have defaults; existing `results.json` files are still deserializable.

---

## 9. Acceptance criteria

- [ ] `InferencePerformanceInfo` includes `p5_time`, `p50_time`, `p95_time`, `p99_time`, `std_time`, `raw_times`.
- [ ] `LatencyMetrics` includes `min_ms`, `std_ms`, `p5_ms`, `max_ms`; p50/p95/p99 map correctly.
- [ ] All 13 accuracy benchmark methods populate `per_example_scores` and `stats`.
- [ ] `ComputePerplexityForDatasetReturn.stats` is a fully-populated `DescriptiveStats`.
- [ ] `results.json` contains `stats` objects inside every benchmark result and inside `latency`.
- [ ] HTML report shows stat cards: Mean, Std Dev, Min, P5, P50, P95, P99, Max for latency.
- [ ] HTML report shows distributional cards for each benchmark (perplexity and accuracy).
- [ ] Latency chart includes P5 and P99 series as reference bounds.

---

## 10. Validation plan

- Run `uv run examples/e2e_option_c_validation.py` with `num_benchmark_runs=5`, `FIRST_N=5`.
- Inspect `compression_results/option-c-first-n-validation/results.json` for presence of `stats` keys.
- Open `report.html` and verify stat cards and distribution sections render correctly.

---

## 11. Rollout plan

1. Create `stats.py` → update type models → update benchmarkers → update adapter → update reporter + templates → update example script.
2. Run end-to-end validation after each layer.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| `num_benchmark_runs=1` produces all-equal stats | Increased to 5 in example script; documented caveat |
| Empty per_example_scores for truthfulqa when dataset returns 0 rows | `DescriptiveStats.from_values([])` returns all-zero model; guarded in template |

---

## 13. Open questions

- *(none — all resolved during planning session)*
