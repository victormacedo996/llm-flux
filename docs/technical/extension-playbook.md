# Extension Playbook

Use this playbook when adding a new concrete class to the framework.

## Goal
Add behavior without changing orchestration logic.

## Standard extension path
1. Identify matching port interface:
   - model loading → `ModelHandle`
   - compression method → `CompressionPort`
   - healing strategy → `HealingPort`
   - profiling strategy → `ProfilingPort`
   - dataset source → `DatasetPort`
2. Create adapter in corresponding folder.
3. Add config class with typed, validated fields.
4. Integrate in an example pipeline under `examples/`.
5. Add a spec document using templates in `docs/specs/`.

## Adapter checklist
- [ ] Class inherits exactly one expected port.
- [ ] Public method contract (`load/compress/heal/profile`) is respected.
- [ ] Error paths are explicit and actionable.
- [ ] Optional dependencies are documented.
- [ ] Outputs are compatible with canonical result models.

## Compression adapter contract checklist
- [ ] Uses `CompressionConfig`-style validated configuration.
- [ ] Raises `CompressionNotSupportedError` for incompatible models.
- [ ] Returns transformed model object for next step.

## Profiling adapter contract checklist
- [ ] Produces `ProfilingResult` with non-empty latency metrics.
- [ ] Stores expensive/raw details in `extra` instead of ad-hoc fields.

## Healing adapter contract checklist
- [ ] Accepts model and returns healed model.
- [ ] Handles dataset source through dataset adapters when possible.

## Dataset adapter checklist
- [ ] Supports `max_samples` semantics.
- [ ] Keeps output Trainer-compatible.

## Example completion criteria
A new adapter is considered integrated when:
1. One `examples/*` script uses it in a full pipeline.
2. The pipeline executes up to at least one profile checkpoint.
3. Result output can be serialized successfully.
