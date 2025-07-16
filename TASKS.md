# May Update Trajectory Tasks

This task list tracks the work needed to add new forward and backward trajectories for the May update and to integrate them into the existing code-base for validation and plotting.

## Completed Tasks

_No tasks completed yet._

## In Progress Tasks

- [ ] Add trajectories to May update (`timelines-refactor`)
  - [ ] Update `get_base_time` to return **(progress, horizon)**
    - [ ] Split into phases
      - [ ] Growth
        - [ ] Exponential to super-intelligence
          - [ ] Phase: Exp
          - [ ] Phase: Super
        - [ ] Phase: Sub
      - [ ] Cost / speed considerations
      - [ ] Release delay? _(confirm whether this phase exists)_
    - [ ] Generate evenly spaced progress points for each phase
    - [ ] Compute horizons using analytical formulas
  - [ ] Implement next function to return **(time, horizon)**

- [ ] Adopt the "main" code to validate against the initial *Eli* forward trajectory plot

- [ ] Add back-cast trajectories to May update
  - [ ] Implement back-cast of `get_base_time` to return **(progress, horizon)** for historical points
    - [ ] Single-phase selection
      - [ ] Phase: Exp
      - [ ] Phase: Sub
    - [ ] Evenly space progress points back a configurable number of progress units
    - [ ] Compute horizons analytically for each historical point
  - [ ] Implement speed-up model to return **(time, horizon)**
    - [ ] Identify and document contributing factors to speed-up _(TODO)_

- [ ] Integrate existing plotting utilities to generate combined trajectory visualisations

## Future Tasks

_Add future enhancements here as they are discovered._

## Implementation Plan

1. Extend data generation utilities in `timelines-refactor/` to support forward and backward trajectory synthesis based on progress/horizon tuples.
2. Refactor plotting code to accept the new trajectory data structures and validate against existing *Eli* plots.
3. Ensure analytical formulas for horizon calculations are well-tested and documented.
4. Iterate on speed-up modelling, calibrating against benchmark datasets.

### Relevant Files

- `timelines-refactor/simple_forecasting_timelines.py` – original timeline generation logic
- `timelines-refactor/simple_forecasting_timelines_may.py` - updated model accounting for labor
- `timelines-refactor/simple_forecasting_timelines_plotting.py` – plotting utilities for trajectories
- `timelines-refactor/` (other modules) – helpers and shared functions
- `timelines/figures/` – destination for generated figures ✅ (once plots are updated) 