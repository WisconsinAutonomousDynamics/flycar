
# A1 — FlyCar Core Profiling

## Objective

Measure the runtime cost of FlyCar's physics, sensor, and environment operations without changing simulation behavior or frozen interfaces.

The goal of A1 is to identify performance bottlenecks and establish a reproducible baseline for future optimization work.

## Implementation

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

### Instrumented Operations

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

### Profiling Output

The profiler returns a dictionary containing:

```python
{
    "timings_s": {...},
    "calls": {...},
    "wall_s": ...,
    "control_steps": ...,
    "simulated_s": ...
}
```

The output is available at:

```python
info["extras"]["core"]["env"]
```

Profiling is optional and disabled by default.

## Benchmark Configuration

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

## Benchmark Results

The following results were recorded from a completed 10-episode run using 20 × 20 CPU raycast eyes.

### Overall Performance

| Metric | Result |
|---|---:|
| Episodes | 10 |
| Total simulated time | 367.050 s |
| Total wall-clock time | 533.006 s |
| Overall real-time factor | 0.689× |

The real-time factor (RTF) is calculated as:

```text
RTF = total simulated time / total wall-clock time
```

An RTF of 1.0× means the simulation runs at real-time speed.

An RTF below 1.0× means the simulation takes longer than the simulated time it advances.

### Per-Episode Results

| Episode | Simulated Time (s) | Wall Time (s) | RTF |
|---|---:|---:|---:|
| 1 | 40.00 | 82.25 | 0.49× |
| 2 | 40.00 | 78.72 | 0.51× |
| 3 | 7.05 | 5.43 | 1.30× |
| 4 | 40.00 | 36.93 | 1.08× |
| 5 | 40.00 | 36.65 | 1.09× |
| 6 | 40.00 | 36.62 | 1.09× |
| 7 | 40.00 | 38.16 | 1.05× |
| 8 | 40.00 | 72.05 | 0.56× |
| 9 | 40.00 | 76.59 | 0.52× |
| 10 | 40.00 | 69.61 | 0.57× |

Episode 3 ended after 7.05 simulated seconds. The remaining episodes reached 40 simulated seconds.

### Phase Timing Breakdown

| Phase | Cumulative Time (s) | Percentage of Wall Time |
|---|---:|---:|
| Chrono physics | 304.1990 | 57.07% |
| Eye rendering | 169.7745 | 31.85% |
| Contact polling | 28.5343 | 5.35% |
| Vehicle synchronization | 4.9052 | 0.92% |
| Observation generation | 4.3954 | 0.82% |
| IMU reading | 3.8543 | 0.72% |
| Vehicle advancement | 3.7513 | 0.70% |
| Info generation | 2.1589 | 0.41% |
| Terrain synchronization | 1.2974 | 0.24% |
| Terrain advancement | 0.6359 | 0.12% |
| Eye updates | 0.2428 | 0.05% |
| World synchronization | 0.1950 | 0.04% |
| World advancement | 0.1701 | 0.03% |
| Contact reading | 0.1275 | 0.02% |

### Main Findings

The two largest measured runtime costs are:

1. **Chrono physics integration:** 57.07% of wall-clock time.
2. **CPU raycast eye rendering:** 31.85% of wall-clock time.

Together, these operations account for approximately 88.9% of the measured wall-clock time.

Contact polling is the next-largest measured category, accounting for approximately 5.35%.

These results suggest that future optimization efforts should initially focus on physics integration and eye rendering.

### Timing Interpretation

Some profiling categories are nested.

For example, IMU reading is measured separately but is also included within observation-generation time.

Therefore, timing percentages should not all be added together.

The reported percentages use total benchmark wall-clock time as the denominator.

Measurements also include effects from the operating system, Python runtime, and machine workload.

## Performance Variability and Limitations

Episode wall-clock times varied substantially.

Some 40-second episodes completed in approximately 37–38 seconds, while others required approximately 70–82 seconds.

This variability may be influenced by factors such as:

- CPU frequency and thermal conditions
- Laptop power-management settings
- Background system activity
- Operating-system scheduling
- Differences in simulation workload

The exact cause of the remaining variability has not been established.

An earlier benchmark run was interrupted when the laptop was closed. That run contained extremely large wall-clock measurements and is not used as the primary performance result.

The results in this report come from the subsequent completed benchmark.

The benchmark measures performance on the machine used for testing and should not be interpreted as a universal FlyCar performance guarantee.

## Validation

### Full Test Suite

Command:

```powershell
python -m pytest -q
```

Result:

```text
100 passed, 4 skipped in 117.76s
```

### Baseline Tests

Command:

```powershell
python -m pytest tests/test_baseline.py -q
```

Result:

```text
2 passed, 4 skipped in 9.42s
```

The baseline tests passed without regenerating snapshots.

### Profiling Tests

Tests were added to `tests/test_core.py` to verify:

- Profiling measurements accumulate across control steps.
- Physics call counts match the expected number of substeps.
- Profiling counters reset between episodes.
- Profiling-enabled and profiling-disabled simulations produce equivalent results for the tested trajectory.

The simulation-equivalence test compares vehicle pose, reward, termination status, truncation status, and IMU output.

These tests provide regression coverage but do not constitute a proof of equivalence for every possible scenario.

## Reproduction

### Requirements

Use the FlyCar project environment with its required Python dependencies, including PyChrono, NumPy, and Gymnasium.

### Run the Benchmark

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

The JSON output is supported by the updated benchmark script; the numerical results documented above were collected before that script revision.

For more reliable timing measurements:

- Keep the computer awake throughout the benchmark.
- Avoid running heavy background applications.
- Use consistent power settings.
- Record hardware and software versions when comparing results across machines.

## Files Changed

| File | Purpose |
|---|---|
| `flycar/config/core.py` | Adds the optional profiling configuration |
| `flycar/env/core.py` | Implements timing instrumentation and profiling output |
| `tests/test_core.py` | Adds profiling regression tests |
| `scripts/benchmark_core.py` | Provides a reproducible benchmark |
| `docs/A1_PROFILING.md` | Documents implementation and results |

## Conclusions

A1 establishes optional runtime profiling for FlyCar's core environment.

The implementation records execution costs across physics, sensors, and environment operations while preserving the existing execution sequence and default behavior.

The benchmark identifies Chrono physics integration and CPU raycast eye rendering as the primary performance costs.

Future optimization work should investigate these components independently while maintaining simulation determinism and existing interface contracts.
