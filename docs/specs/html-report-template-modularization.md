# HTML Report Template Modularization

## Metadata
- Spec ID: `SPEC-0001`
- Title: Modularize HTML report template with nested report partials
- Author: GitHub Copilot
- Date: 2026-03-31
- Status: `validated`
- Related decisions: `DEC-006`

## 1. Problem statement
The current HTML report template is implemented as a single large Jinja file that mixes page structure, styling, section markup, and imperative client-side rendering code. This increases friction when adding a new report section or refactoring an existing one, and it makes the template harder to work with in AI-assisted editing flows where context window size matters. The report must remain visually and structurally equivalent after refactoring.

## 2. Goals
- Reduce maintenance friction by moving the report template to a nested `report/` folder with modular partials.
- Preserve rendered visual output and DOM structure for the generated report.
- Keep the existing Python report data contract unchanged during this refactor.
- Document every structural change so future edits can follow the same pattern.

## 3. Non-goals
- Changing the report data model or JSON contract passed from `llm_flux.core.html_reporter`.
- Rewriting inline JavaScript behavior or chart logic beyond structural extraction.
- Introducing external CSS or JavaScript assets.

## 4. Scope
### In scope
- Move the report entry template under `llm_flux/core/templates/report/`.
- Extract reusable partials and macros for layout, sections, and scripts.
- Update the template loader call site in `llm_flux/core/html_reporter.py`.
- Add documentation for the new template layout and maintenance rules.

### Out of scope
- Redesigning the report UI.
- Changing report generation APIs.
- Adding new report sections or new metrics.

## 5. User scenarios
- A maintainer adds a new report section without touching unrelated chart or table markup.
- A maintainer updates a single section of the report without loading the full template into editor/LLM context.
- A contributor understands the report template layout from project documentation and local maintainer notes.

## 6. Proposed solution
Create a nested `llm_flux/core/templates/report/` template tree with a single composed entry template and focused partials for head/styles, header, DAG, hardware, charts, metrics table, details panel, footer, and scripts. Use Jinja includes and macros to preserve the current HTML order and DOM identifiers. Update `llm_flux.core.html_reporter` to load the new entry template path while keeping the same render context.

## 7. Decision choices
- Choice: Template modularization strategy.
- Alternatives:
  - Keep a single template file and only add comments.
  - Move JavaScript/CSS into external assets.
  - Split the template into nested partials under `report/`.
- Selected option:
  - Split the template into nested partials under `report/`.
- Rationale:
  - Lowers maintenance friction without changing report behavior.
  - Keeps the report self-contained.
  - Reduces context-window pressure for future AI-assisted edits.
- Trade-offs:
  - More files to manage.
  - Template include order must be preserved carefully.

- Choice: Output compatibility target.
- Alternatives:
  - Allow visual-only parity.
  - Require visual and DOM equivalence.
- Selected option:
  - Require visual and DOM equivalence.
- Rationale:
  - Minimizes regression risk for existing consumers and screenshots.
- Trade-offs:
  - Limits cleanup opportunities during the refactor.

- Choice: Data contract scope.
- Alternatives:
  - Reshape the render context during the refactor.
  - Keep the current render context unchanged.
- Selected option:
  - Keep the current render context unchanged.
- Rationale:
  - Isolates the change to the templating layer.
- Trade-offs:
  - Some presentational shaping remains in client-side JavaScript.

## 8. Technical impact
- Modules affected:
  - `llm_flux/core/html_reporter.py`
  - `llm_flux/core/templates/report/*`
  - `docs/technical/architecture.md`
  - `docs/product/decision-log.md`
- New dependencies:
  - None.
- Data model changes:
  - None.
- Backward compatibility notes:
  - Generated report markup, DOM ids/classes, conditional sections, and client-side behavior must remain equivalent.

## 9. Acceptance criteria (testable)
- [x] The report template entrypoint lives under `llm_flux/core/templates/report/`.
- [x] The report is composed from modular partials and/or macros grouped by concern.
- [x] `llm_flux.core.html_reporter` renders the new entry template path without changing the render context contract.
- [x] Generated report sections preserve the same DOM ids/classes and visual behavior as before.
- [x] Documentation describes the new template layout and how to extend it.
- [x] The spec and decision log record the structural changes and constraints.

## 10. Validation plan
- Unit/integration checks:
  - Ran targeted Python error checks on `llm_flux/core/html_reporter.py` with no errors reported.
  - Rendered a representative report after the refactor and confirmed the nested template path and includes load correctly.
  - Compared rendered output against the previous committed monolithic template and confirmed heading order and DOM id sequences are unchanged.
- Example execution path:
  - Generated an HTML report from a representative `PipelineRunResult` built from `ProfilingResult`, `LatencyMetrics`, `MemoryMetrics`, and `AccuracyMetrics`.
- Expected artifacts:
  - A generated HTML report with unchanged visible sections and DOM anchors.
  - Updated documentation and decision log.

## 11. Rollout plan
- Step 1: Add spec and decision entry.
- Step 2: Move the report entry template into `report/` and extract modular partials.
- Step 3: Update the Python template lookup path.
- Step 4: Update technical documentation and local maintainer notes.
- Step 5: Validate rendering and mark the spec as implemented.

## 12. Risks and mitigations
- Risk:
  - Partial extraction changes script execution order or DOM anchors.
- Mitigation:
  - Keep section order, ids/classes, and inline script order unchanged.

- Risk:
  - Include boundaries introduce whitespace or markup drift.
- Mitigation:
  - Preserve markup blocks nearly verbatim inside the extracted partials.

- Risk:
  - Future contributors edit the wrong partial.
- Mitigation:
  - Add maintainer-facing documentation that maps sections to files.

## 13. Open questions
- None.

## 14. Implementation results
- The monolithic template was replaced with a nested `report/` template tree composed from section partials and ordered script partials.
- `llm_flux/core/html_reporter.py` now resolves `report/html_reporter.html.jinja2` as the single report entrypoint.
- Maintainer-facing documentation was added to the report template folder and technical architecture docs.
