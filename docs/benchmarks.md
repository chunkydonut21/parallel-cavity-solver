# Benchmarks on my laptop

These are local measurements, not numbers built into the browser page. The
machine has an Intel Core Ultra 9 275HX and runs Windows 11 Home. I used MSVC
Release builds. I did not pin workers to cores or control background load, so
the tables show what happened in these runs rather than a machine-wide speed
limit. Each comparison below has its own serial run; its serial time is the
denominator for that table only.

## Poisson solver as the grid grows

The standalone benchmark starts from zero and solves a manufactured *discrete*
problem with a known answer. All three sizes use the same maximum equation
residual of `1e-6`. Each number below is the median of five solves.

| Grid | Jacobi sweeps | Median time | Maximum solution error | Cell updates/s |
| ---: | ---: | ---: | ---: | ---: |
| 33 × 33 | 3,480 | 0.002919 s | 5.07 × 10⁻⁸ | 1.146 × 10⁹ |
| 65 × 65 | 13,940 | 0.050181 s | 5.05 × 10⁻⁸ | 1.103 × 10⁹ |
| 129 × 129 | 55,780 | 0.816407 s | 5.04 × 10⁻⁸ | 1.102 × 10⁹ |

Doubling the intervals gives roughly four times as many interior cells and
four times as many Jacobi sweeps at this residual target. That is about sixteen
times as many cell updates, and the measured time grew at about that rate.
The update rate stayed close to 1.1 billion cells/s. This is why a faster
sweep alone will not fix Jacobi's iteration cost.

```powershell
.\build\Release\poisson_serial_bench.exe --grid 33 --repeats 5 --output results\poisson_33.csv
.\build\Release\poisson_serial_bench.exe --grid 65 --repeats 5 --output results\poisson_65.csv
.\build\Release\poisson_serial_bench.exe --grid 129 --repeats 5 --max-iters 100000 --output results\poisson_129.csv
.\.venv\Scripts\python.exe scripts\plot_poisson.py results\poisson_33.csv results\poisson_65.csv results\poisson_129.csv
```

The larger case needs a higher iteration cap; the stopping residual is still
`1e-6`. The browser benchmark batches short solves to make sub-millisecond
cases easier to time, then shows the median time per solve from three batches.

## Poisson cost inside the cavity

A serial 65 × 65, Re=100 cavity run took 4,000 timesteps and 66.29988 s of
numerical time. Poisson solves used 66.234225 s of that; vorticity transport
used 0.0443011 s. The run averaged 4371.9052 Jacobi sweeps per timestep.
Its final maximum vorticity change rate was 0.10704716, so this was a
transient run. It shows where the time went, not a validated steady result.

```powershell
.\build\Release\cavity_serial.exe --grid 65 --steps 4000 --output results\cavity_instrumented.csv --metrics results\cavity_instrumented_metrics.json
```

The steady run used the same grid and Reynolds number, with a stopping target
of `max |Δω|/dt ≤ 0.001` after at least 4,000 timesteps. It was measured on
6 October 2026 with MSVC 19.51.36260.0 in Release mode.

| Item | Value |
| --- | ---: |
| Completed timesteps | 10,547 |
| Timestep | 0.001220703125 |
| Simulated time | 12.874756 |
| Numerical runtime | 114.71828 s |
| Mean Jacobi sweeps per timestep | 2645.3703 |
| Final Poisson residual | 9.9690059 × 10⁻⁷ |
| Final vorticity change rate | 9.993924 × 10⁻⁴ |

The validator found a maximum discrete interior divergence of `7.959 ×
10⁻¹⁵`. The primary vortex was at grid point `(0.60938, 0.73438)`, about
0.00783 from the Re=100 reference centre `(0.6172, 0.7344)` reported by
[Ghia, Ghia & Shin (1982)](https://doi.org/10.1016/0021-9991(82)90058-4).
That centre check is useful, but I have not yet compared full centerline
velocity profiles or repeated the validation at several grid sizes.

```powershell
.\build\Release\cavity_serial.exe --grid 65 --steps 20000 --steady-tol 0.001 --min-steps 4000 --output results\cavity_steady.csv --metrics results\cavity_steady_metrics.json
.\.venv\Scripts\python.exe scripts\validate.py --input results\cavity_steady.csv --metrics results\cavity_steady_metrics.json
```

## OpenMP

This comparison used 65 × 65 points, Re=100, and 100 timesteps. Each setup
ran three times. The table gives the median numerical runtime. File output
is outside the timer.

| Version | Threads | Median time | Speedup vs serial | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Serial | 1 | 2.2581 s | 1.00× | — |
| OpenMP | 1 | 2.5472 s | 0.89× | 88.7% |
| OpenMP | 2 | 1.6785 s | 1.35× | 67.3% |
| OpenMP | 4 | 1.2593 s | 1.79× | 44.8% |

One OpenMP thread was slower than the serial build: it still enters OpenMP
regions and combines per-row maxima. Four threads helped, though speedup was
far from fourfold. All saved `x`, `y`, `u`, `v`, `speed`, `omega`, and `psi`
values matched the serial run exactly here. The script also rejects a field
difference above `1e-10`. Exact equality in this case is an observation;
parallel floating-point code can produce small differences in general.

```powershell
.\.venv\Scripts\python.exe scripts\bench_openmp.py --grid 65 --steps 100 --repeats 3 --threads 1 2 4 --output-dir results\openmp_bench_65_100
```

## MPI

The MPI comparison used the same 65 × 65, Re=100, 100-step *workload* and
three repeats per setup. It was a separate run, with its own serial baseline.
MPI time includes row exchange and global reductions. Process startup, final
field gathering, and file output are outside the timer.

| Version | Processes | Median time | Speedup vs serial | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Serial | 1 | 2.3420 s | 1.00× | — |
| MPI | 1 | 2.9602 s | 0.79× | 79.1% |
| MPI | 2 | 2.0459 s | 1.14× | 57.2% |
| MPI | 4 | 1.5006 s | 1.56× | 39.0% |

One MPI rank still uses the distributed solver and MPI calls, which cost
time without parallel work. Four ranks reduced runtime, but each Jacobi sweep
also needed ghost-row exchange, and every residual check needed a global
maximum. These runs do not separate the costs or say how a cluster would
behave. All saved fields matched serial exactly. The automated test also
checks 1, 2, and 4 ranks on a 17 × 17 grid, including an uneven 4/4/4/3 row
split. Validation passed for the four-rank 65 × 65 output; its steady-flow
reference check was skipped because 100 steps did not reach steady flow.

```powershell
.\.venv\Scripts\python.exe scripts\bench_mpi.py --grid 65 --steps 100 --repeats 3 --ranks 1 2 4 --output-dir results\mpi_bench_65_100
```

Both scripts save raw runs and `summary.csv` under `results/`. CTest runs
smaller serial/OpenMP and serial/MPI equality checks as part of the build.

## The next measurement I want

The 65 × 65 case has only 63 interior rows. On a 24-core CPU, it is too small
to show where a larger computation stops benefiting from more workers. I want
to repeat these tests at larger grids and thread/rank counts, with worker
placement and background load recorded. For MPI, I also want separate timers
for stencil work, row exchange, and reductions. A useful first model is

`time per sweep ≈ cell work / workers + row exchange + convergence check`.

The current tables tell me what happened. Those additional measurements would
let me test *why* it happened.
