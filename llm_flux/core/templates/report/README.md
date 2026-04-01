# Report Templates

This folder contains the modular HTML reporter templates introduced by `SPEC-0001` and `DEC-006`.

## Layout
- `html_reporter.html.jinja2` — composed report entrypoint loaded by `llm_flux.core.html_reporter`.
- `macros.html.jinja2` — shared Jinja macros for repeated layout atoms.
- `partials/` — section-level template fragments.
- `partials/scripts/` — JavaScript fragments kept in execution order inside a single `<script>` block.

## Maintenance rules
- Preserve report section order unless an approved spec explicitly changes it.
- Preserve DOM ids/classes consumed by styling and JavaScript.
- Preserve script execution order when editing files under `partials/scripts/`.
- Keep the Python render-context contract unchanged unless a follow-up spec approves data-model changes.
- Document non-trivial report template changes in `docs/specs/`, `docs/product/decision-log.md`, and `docs/technical/architecture.md`.
