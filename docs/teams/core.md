# Team: Core & body sensing (`core`)

This page belongs to the team. Keep it up to date here instead of in the README.
The Phase 1 work guide (tasks, parts A to C) is [`teams/team-core.md`](../../teams/team-core.md).

## What you own

| Path | What it is |
|---|---|
| `flycar/env/core.py` | `FlyCarEnv`: `reset()`, `step()`, the Chrono system and the time loop |
| `flycar/env/sensing/imu.py` | Speed, yaw rate, acceleration |
| `flycar/env/sensing/contact.py` | Left / front / right touch flags |
| `flycar/config/core.py` | Scenario sections `episode`, `timing`, `contact`, `termination`: defaults and checks |
| `tests/test_core.py` | Your tests |
| `scenarios/team_core.yaml` | Your scratch scenario |
| `flycar/drivers/contrib/core_driver.py` | Your capstone driver (observation only, no pychrono) |
| `teams/team-core.md` | Your Phase 1 work guide |
| `docs/teams/core.md` | This page |

## Contracts you must not break

Changing these needs a lead-approved `api-change` issue (see
[`docs/CONTRACTS.md`](../CONTRACTS.md)):

- **Everything in `flycar/env/spaces.py`** (contract 1): the action, the
  observation (`imu`, `contact`, `time` ...) and the info fields. Only the
  *contents* of `info["extras"]` are free.
- **The step order in `_physics_step()`** (contract 7): terrain, world,
  vehicle, window; synchronize before advance; then
  `system.DoStepDynamics`, then `eyes.update()` and `contact.poll()`.
- Determinism: same seed + same actions = the same trajectory, bit for bit.

You depend on: the `FlyCarVehicle` accessors, `CHASSIS_TAG` and
`WORLD_TAG_BASE`, the eyes interface, and `Terrain`.

## Your reserved config

```yaml
contact:
  mode: flag            # planned: force
termination:
  flipped: false        # planned: end the episode when the car rolls over
  stuck_s: null         # planned: end it after this long without moving
  left_arena: false     # planned: end it when the car leaves the arena
```

Any other value is rejected with "not implemented yet" until you implement it.

## Things to know

- `info["extras"]` is collected in `_extras()` from every component that has
  an `extra_info()` method. Add one to `Imu` or `ContactSensor` to expose,
  say, contact forces (`info["extras"]["core"]`).
- `tests/test_baseline.py` must keep passing: default behaviour may not change.

## Running your tests

```
pytest tests/test_core.py tests/test_baseline.py
python run.py --scenario scenarios/team_core.yaml --driver braitenberg --dashboard
```


## A1 — Core Profiling

A1 implements optional wall-clock profiling of FlyCar's core environment.

- Configuration: `timing.profile`, disabled by default.
- Output: `info["extras"]["core"]["env"]`.
- Measures physics, synchronization, advancement, vision, contact,
  IMU, observation, and info generation.
- Preserves the existing physics execution order and default behavior.
- Resets profiling measurements between episodes.


The updated 10-episode, 20×20 CPU raycast benchmark recorded an overall real-time factor of 1.452×, compared with 0.689× in the previous run.

The new run completed 367.050 simulated seconds in 252.856 wall-clock seconds. Full-length episodes took approximately 25.81–30.53 seconds, showing substantially less variability than the earlier measurements.

Latest core validation: 23 passed.
Latest baseline validation: 2 passed, 4 skipped.
Full-suite validation pending.

The detailed methodology and results are included below.

## A1 — Detailed Profiling Report

### FlyCar Core Profiling

### Objective

Measure the runtime cost of FlyCar's physics, sensor, and environment operations without changing simulation behavior or frozen interfaces.

The goal of A1 is to identify performance bottlenecks and establish a reproducible baseline for future optimization work.

### Implementation

Profiling is controlled by the scenario setting:

```yaml
timing:
  profile: true
```

The default is `false`, preserving existing behavior.

When enabled, profiling information is exposed through:

```python
info["extras"]["core"]["env"]
```

The profiler records:

- Cumulative wall-clock time per operation, in seconds
- Number of calls per operation
- Number of control steps
- Total simulated time
- Total measured episode wall-clock time

#### Instrumented Operations

The following operations are measured:

| Category | Operations |
|---|---|
| Synchronization | Terrain, world, vehicle, window |
| Advancement | Terrain, world, vehicle, window |
| Physics | Chrono `DoStepDynamics()` |
| Vision | Eye updates and rendering |
| Contact sensing | Contact polling and reading |
| IMU | IMU sensor reading |
| Environment | Observation and info generation |

Timing measurements use Python's `time.perf_counter()`.

The instrumentation preserves the original order of operations in `_physics_step()`.

Profiling counters reset between episodes. Initial suspension settling and reset-time sensor initialization are excluded from episode measurements.

#### Profiling Output

The profiler returns a dictionary containing:

```python
{
    "timings_s": {...},
    "calls": {...},
    "wall_s": ...,
    "bookkeeping_overhead_s": ...,
    "control_steps": ...,
    "simulated_s": ...
}
```

The output is available at:

```python
info["extras"]["core"]["env"]
```

Profiling is optional and disabled by default.

### Benchmark Configuration

The benchmark uses the following configuration:

| Setting | Value |
|---|---|
| Scenario | `scenarios/light_seek.yaml` |
| Episodes | 10 |
| Random seeds | 0–9 |
| Eye backend | CPU raycast |
| Eye resolution | 20 × 20 |
| Control action | `[0.3, 0.0, 0.0]` |
| Control period | 0.01 simulated seconds |
| Maximum episode duration | 40 simulated seconds |
| Maximum control steps | 4,000 per episode |
| Early termination | Enabled |

The environment is reused between episodes and reset with a different seed.

Wall-clock measurements begin after `env.reset()` completes and stop when the episode terminates, truncates, or reaches the maximum step count.

Environment construction, reset operations, and initial suspension settling are excluded from the benchmark wall-clock measurement.

### Benchmark Results

The following results were recorded from a completed 10-episode run using 20 × 20 CPU raycast eyes.

#### Overall Performance


| Metric | Result |
|---|---:|
| Episodes | 10 |
| Total simulated time | 367.050 s |
| Total wall-clock time | 252.856 s |
| Overall real-time factor | 1.452× |
| Previous real-time factor | 0.689× |

The real-time factor (RTF) is calculated as:

```text
RTF = total simulated time / total wall-clock time
```

An RTF of 1.0× means the simulation runs at real-time speed.

An RTF below 1.0× means the simulation takes longer than the simulated time it advances.

#### Per-Episode Results


| Episode | Simulated Time (s) | Wall Time (s) | RTF |
|---|---:|---:|---:|
| 1 | 40.00 | 26.63 | 1.50× |
| 2 | 40.00 | 26.58 | 1.50× |
| 3 | 7.05 | 3.95 | 1.78× |
| 4 | 40.00 | 28.03 | 1.43× |
| 5 | 40.00 | 28.34 | 1.41× |
| 6 | 40.00 | 28.54 | 1.40× |
| 7 | 40.00 | 30.53 | 1.31× |
| 8 | 40.00 | 25.81 | 1.55× |
| 9 | 40.00 | 27.91 | 1.43× |
| 10 | 40.00 | 26.54 | 1.51× |

Episode 3 ended after 7.05 simulated seconds. The remaining episodes reached 40 simulated seconds.

#### Phase Timing Breakdown


| Phase | Cumulative Time (s) | Percentage of Wall Time |
|---|---:|---:|
| Chrono physics | 146.4163 | 57.91% |
| Eye rendering | 82.7995 | 32.75% |
| Contact polling | 10.9027 | 4.31% |
| Observation generation | 1.8663 | 0.74% |
| Vehicle synchronization | 1.6755 | 0.66% |
| IMU reading | 1.6158 | 0.64% |
| Vehicle advancement | 1.3591 | 0.54% |
| Info generation | 0.9454 | 0.37% |
| Terrain synchronization | 0.4096 | 0.16% |
| Terrain advancement | 0.2582 | 0.10% |
| Eye updates | 0.0936 | 0.04% |
| World synchronization | 0.0913 | 0.04% |
| World advancement | 0.0778 | 0.03% |
| Contact reading | 0.0504 | 0.02% |


#### Profiling Instrumentation Overhead

Profiling introduces additional work through high-resolution timer calls and the bookkeeping required to accumulate timing measurements.

The updated benchmark separately reports:

| Component | Time (s) | Percentage of Wall Time |
|---|---:|---:|
| Estimated timer-call overhead | 0.4839 | 0.19% |
| Measured bookkeeping overhead | 1.2955 | 0.51% |

Timer-call overhead is estimated using a repeated-call calibration of `time.perf_counter()`. This calibration also includes Python loop overhead and therefore does not isolate the timer cost perfectly.

Bookkeeping overhead is measured around updates to the profiler's timing and call-count dictionaries.

These measurements characterize individual instrumentation costs. They do not constitute a direct measurement of the complete performance difference between profiling-enabled and profiling-disabled execution.

The benchmark's real-time factor includes profiling overhead.

#### Main Findings


#### Main Findings

The two largest measured runtime costs are:

1. **Chrono physics integration:** 57.91% of wall-clock time.
2. **CPU raycast eye rendering:** 32.75% of wall-clock time.

Together, these operations account for approximately 90.66% of the measured wall-clock time.

Contact polling is the next-largest measured category, accounting for approximately 4.31%.

The updated benchmark achieved an overall real-time factor of 1.452×, compared with 0.689× in the previous run.

These results suggest that future optimization efforts should initially focus on physics integration and eye rendering.

#### Timing Interpretation

Some profiling categories are nested.

For example, IMU reading is measured separately but is also included within observation-generation time.

Therefore, timing percentages should not all be added together.

The reported percentages use total benchmark wall-clock time as the denominator.

Measurements also include effects from the operating system, Python runtime, and machine workload.

### Performance Variability and Limitations


### Performance Variability and Limitations

The updated benchmark showed substantially less wall-clock variability than the previous run.

For episodes reaching 40 simulated seconds, wall-clock times ranged from 25.81 to 30.53 seconds. The previous benchmark recorded times ranging from approximately 37 to 82 seconds for comparable full-length episodes.

The updated run achieved an overall real-time factor of 1.452×, compared with 0.689× previously.

The improved consistency suggests that the earlier measurements were affected by substantial performance variability. However, the exact cause of the difference between runs has not been established.

Potential influences include:

- CPU frequency and thermal conditions
- Laptop power-management settings
- Background system activity
- Operating-system scheduling
- Differences in simulation workload

Episode 3 terminated after 7.05 simulated seconds and is therefore not directly comparable to the full-length episodes.

An earlier benchmark run was interrupted when the laptop was closed. That run is excluded from the reported results.

The benchmark measures performance on the machine used for testing and should not be interpreted as a universal FlyCar performance guarantee.

### Validation

#### Core Tests

Command:

```powershell
python -m pytest tests/test_core.py -q
```

Result:

```text
23 passed in 84.66s
```

#### Baseline Tests

Command:

```powershell
python -m pytest tests/test_baseline.py -q
```

Result:

```text
2 passed, 4 skipped in 5.85s
```

The baseline tests passed without regenerating snapshots.

The full test suite has not yet been rerun after the latest profiling changes.

#### Profiling Tests

Tests were added to `tests/test_core.py` to verify:

- Profiling measurements accumulate across control steps.
- Physics call counts match the expected number of substeps.
- Profiling counters reset between episodes.
- Profiling-enabled and profiling-disabled simulations produce equivalent results for the tested trajectory.

The simulation-equivalence test compares vehicle pose, reward, termination status, truncation status, and IMU output.

These tests provide regression coverage but do not constitute a proof of equivalence for every possible scenario.

### Reproduction

#### Requirements

Use the FlyCar project environment with its required Python dependencies, including PyChrono, NumPy, and Gymnasium.

#### Run the Benchmark

From the repository root:

```powershell
conda activate flycar
python -m scripts.benchmark_core
```

The benchmark script:

1. Loads `scenarios/light_seek.yaml`.
2. Enables profiling.
3. Configures 20 × 20 CPU raycast eyes.
4. Runs 10 episodes using seeds 0–9.
5. Applies the fixed action `[0.3, 0.0, 0.0]`.
6. Collects per-phase timing measurements.
7. Reports simulated time, wall time, and real-time factor.
8. Writes results to `benchmark_core_results.json`.


The reported numerical results were generated using the updated benchmark script and saved to `benchmark_core_results.json`.

The benchmark also reports an estimated timer-call overhead and a measured bookkeeping overhead. These measurements are reported separately and do not represent an exact measurement of total profiling overhead.

For more reliable timing measurements:

- Keep the computer awake throughout the benchmark.
- Avoid running heavy background applications.
- Use consistent power settings.
- Record hardware and software versions when comparing results across machines.

### Files Changed

| File | Purpose |
|---|---|
| `flycar/config/core.py` | Adds the optional profiling configuration |
| `flycar/env/core.py` | Implements timing instrumentation and profiling output |
| `tests/test_core.py` | Adds profiling regression tests |
| `scripts/benchmark_core.py` | Provides a reproducible benchmark |
| `docs/teams/core.md` | Documents implementation and results |


### Conclusions

A1 establishes optional runtime profiling for FlyCar's core environment while preserving the existing physics execution order and default behavior.

The updated 10-episode benchmark completed 367.050 simulated seconds in 252.856 wall-clock seconds, achieving an overall real-time factor of 1.452×.

Chrono physics integration (57.91%) and CPU raycast eye rendering (32.75%) remain the primary performance bottlenecks, accounting for approximately 90.66% of measured wall-clock time.

The profiler additionally reports estimated timer-call overhead of 0.19% and measured bookkeeping overhead of 0.51% of benchmark wall time. These are separate instrumentation measurements rather than an exact total profiling overhead.

Compared with the previous benchmark's 0.689× real-time factor, the updated run demonstrates substantially improved performance and consistency under the measured conditions.

Future optimization work should prioritize physics integration and eye rendering while maintaining simulation determinism and existing interface contracts.