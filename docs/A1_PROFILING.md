
# A1 — FlyCar Core Profiling

## Objective

Measure the runtime cost of FlyCar's physics, sensor, and environment operations without changing simulation behavior or frozen interfaces.

## Implementation

Profiling is controlled by `timing.profile`, which defaults to `false`.

When enabled, timing information is exposed through:

`info["extras"]["core"]["env"]`

The profiler records:
- Cumulative wall-clock time per operation
- Number of calls per operation
- Control steps and simulated time
- Total measured episode wall time

Instrumented operations include:
- Terrain, world, and vehicle synchronization
- Terrain, world, and vehicle advancement
- Chrono `DoStepDynamics`
- Eye updates and rendering
- Contact polling and reading
- IMU reading
- Observation and info generation

Profiling counters reset between episodes. Suspension settling is excluded from episode measurements.

## Benchmark Configuration

- Scenario: `scenarios/light_seek.yaml`
- Episodes: 10
- Seeds: 0–9
- Eye backend: CPU raycast
- Eye resolution: 20 × 20
- Control action: `[0.3, 0.0, 0.0]`
- Maximum episode duration: 40 simulated seconds
- Early termination: Enabled
- Setup and reset time: Excluded

## Benchmark Results

| Metric | Result |
|---|---:|
| Episodes | 10 |
| Simulated time | 367.050 s |
| Wall-clock time | 533.006 s |
| Real-time factor | 0.689× |
| Chrono physics | 304.199 s (57.07%) |
| Eye rendering | 169.775 s (31.85%) |
| Contact polling | 28.534 s (5.35%) |

Physics integration and CPU eye rendering account for approximately 88.9% of measured wall time.

Timing categories may overlap, so percentages should not all be summed.

## Performance Variability

Episode runtimes varied substantially. Some 40-second episodes completed in approximately 37–38 seconds, while others required 70–82 seconds.

An earlier benchmark was interrupted when the laptop was closed and is not used as the primary performance result.

The cause of the remaining runtime variability has not been established.

## Validation

Full test suite:

`python -m pytest -q`

Result:

`100 passed, 4 skipped in 117.76s`

Tests include profiler accumulation, reset behavior, and comparison of simulation outputs with profiling enabled and disabled.

## Reproduction

From the repository root:

```powershell
conda activate flycar
python -m scripts.benchmark_core
```

The benchmark uses the configured scenario with profiling enabled, 20 × 20 raycast eyes, and seeds 0–9.

For reliable wall-clock measurements, keep the machine awake and avoid heavy background workloads.

## Conclusions

The profiling implementation identifies Chrono physics integration and CPU eye rendering as the primary runtime costs.

Future optimization work should investigate these components separately while preserving simulation determinism and existing interfaces.
