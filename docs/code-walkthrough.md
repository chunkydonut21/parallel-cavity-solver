# Code notes

This page is a map of the implementation. The calculation starts with a flat
grid of doubles, runs a Jacobi Poisson solve inside each cavity timestep, and
then updates vorticity. OpenMP and MPI divide that work in different ways.

## Build targets

`CMakeLists.txt` builds four executables:

| Target | Main source | Use |
| --- | --- | --- |
| `cavity_serial` | `src/main.cpp` | Serial cavity run |
| `cavity_openmp` | `src/main.cpp` | Same code with OpenMP loops enabled |
| `cavity_mpi` | `src/mpi_main.cpp` | Cavity run split between MPI processes |
| `poisson_serial_bench` | `src/poisson_bench.cpp` | Poisson solve with a known answer |

Both serial and OpenMP compile `src/cavity.cpp`. The OpenMP target defines
`CAVITY_USE_OPENMP`; `#ifdef` blocks add pragmas only to loops whose grid
updates are independent. MPI uses `src/mpi_solver.cpp` because each process
holds only its own strip of the global grid. CMake finds Microsoft MPI on
Windows and uses `find_package(MPI)` elsewhere.

The browser dependencies are in `requirements-web.txt`. Build files, output,
and the Python virtual environment stay out of Git through `.gitignore`.

## The serial grid and settings

`src/cavity.hpp` defines `Grid`, `Config`, `PoissonResult`, `RunSummary`, and
`Solver`. `Grid` stores `N × N` values in one `std::vector<double>`. Accessing
`grid(i, j)` uses position `j*N+i`, so adjacent columns are next to each
other in memory. Two `operator()` overloads allow both reading a `const`
grid and changing a writable one. `swap()` exchanges two grids without
copying their cells; `fill()` resets one.

`Config` contains requested settings such as grid size, Reynolds number,
Poisson tolerance, timestep limit, and output paths. `RunSummary` holds what
the run actually did: completed steps, chosen timestep, runtime and kernel
times, final residual, and steady-stop status. `PoissonResult` is smaller:
it holds the sweep count and residual from one Poisson solve.

`Solver` owns separate grids for `psi` (streamfunction), `omega` (vorticity),
`u` and `v` (velocity), and temporary grids for new `psi` and `omega` values.
One cavity timestep can contain thousands of Jacobi *sweeps*. A small Poisson
residual means the streamfunction solves its grid equation; it does not mean
the cavity flow has stopped changing.

## One cavity timestep

The numerical code is in `src/cavity.cpp`. `Solver::run()` repeats this order:

```text
solve Poisson for psi → calculate velocity → set wall vorticity → advance omega
```

`solve_poisson()` uses a five-point finite-difference stencil for
`∇²ψ = −ω`. At each interior cell it calculates:

```text
new psi = (left psi + right psi + lower psi + upper psi + h² × omega) / 4
```

It reads only the old `psi` grid and writes each result to `scratch`. Once
the whole sweep is finished, the grids are swapped. Updating `psi` in place
would make later cells read partly new values, changing the algorithm and
making the loop harder to parallelize. `max_poisson_residual()` calculates
the largest absolute error in the discrete equation. The code checks after
the first sweep, then every 20 sweeps, and stops when the requested residual
is reached. It throws an error if the iteration cap is reached first.

`choose_timestep()` uses grid spacing, lid speed, and viscosity to select a
conservative fixed value unless the user passes `--dt`. Here viscosity is
`ν = U L / Re`. `compute_velocity()` takes central differences of `psi` for
the interior and sets the moving top-wall speed explicitly.

`apply_vorticity_boundary()` uses nearby streamfunction values to set wall
vorticity. The top wall formula includes its speed. The two top corners are
assigned to the stationary side walls.

`advance_vorticity()` reads the old `omega` grid and writes to `omega_next_`.
It chooses the upstream neighbour for each advection difference, uses a
centered Laplacian for diffusion, and takes an explicit Euler timestep. It
tracks the largest `|new omega - old omega|/dt` for optional steady stopping.
Only after all interior cells are updated does it swap the grids. It also
stops if an update becomes `NaN` or infinity.

`run()` times the major parts with `std::chrono`. After the last timestep,
it solves Poisson and calculates velocity again so the CSV fields describe
the same final state. `save_csv()` writes one row per cell; `save_metrics()`
writes the run information as JSON. File output is outside the numerical
timer. `src/main.cpp` handles command-line options and calls these methods.

## A Poisson case with a known answer

`src/poisson_bench.cpp` checks the solver independently of cavity flow. It
starts with `sin(πx)sin(πy)` as a known streamfunction, applies the *discrete*
five-point Laplacian, and builds the matching `omega`. Starting from zero,
Jacobi should recover the known grid field. The executable records both the
equation residual and maximum error against that field.

Every repeat starts from zero. `--batch` times several fresh solves together
and reports time per solve; this reduces timer noise for small grids.
`scripts/plot_poisson.py` plots the resulting sweep counts and timings.

## What OpenMP changes

In the OpenMP build, `#pragma omp parallel for schedule(static)` splits rows
of the Jacobi update between software threads. Every thread reads the old
`psi`; each writes different cells in `scratch`. The loop finishes before
the grids are swapped. Velocity reconstruction and vorticity transport use
the same pattern: shared input grid, separate output cells.

The residual and vorticity-change checks need one maximum for the whole grid.
Each worker writes a maximum for its row, and the program combines those
values after the loop. The MSVC OpenMP mode used here does not support the
`max` reduction clause. `src/main.cpp` accepts `--threads N` in this build
and disables dynamic thread-count changes for benchmark runs.

OpenMP threads share one address space. A thread can read a neighbouring row
directly, so this version needs no ghost rows or messages.

## What MPI changes

`src/mpi_solver.hpp` declares `LocalGrid` and `MpiSolver`.
`src/mpi_solver.cpp` implements them. Each MPI process, called a *rank*, owns
a contiguous strip of interior rows. `LocalGrid` holds those rows plus one
ghost row above and below. With `N=17` and four ranks, the 15 interior rows
split as 4, 4, 4, and 3. The earlier ranks take any remainder.

`exchange_rows()` uses `MPI_Sendrecv` to copy edge rows between neighbours.
Before a Jacobi sweep, each rank updates its ghost `psi` rows and then
calculates only its owned cells. The physical top and bottom ranks use
`MPI_PROC_NULL` where no neighbour exists. A ghost row is a copy used for
reading; the receiving rank does not update it as an owned row.

After a sweep, a residual check needs fresh ghost rows again. Each rank
calculates its local maximum, then `MPI_Allreduce` gives every rank the
largest residual anywhere on the grid. They must all reach the same stop
decision. The vorticity update exchanges `omega` rows and similarly reduces
its maximum change and non-finite flag.

`MpiSolver::run()` follows the serial timestep order. `MPI_Barrier` lines up
the start of timing, `MPI_Wtime()` measures each rank, and another reduction
reports the slowest rank's elapsed time. That includes message exchange and
reductions. After the timed simulation, `MPI_Gatherv` collects completed
fields at rank 0 for output; the actual calculation was distributed.

`src/mpi_main.cpp` parses options, calls `MPI_Init`, starts `MpiSolver` on
every rank, and ends with `MPI_Finalize`. If one rank fails, `MPI_Abort` stops
the other ranks so they do not wait indefinitely in an MPI call.

## Checks and scripts

`tests/test_solver.cpp` checks the manufactured Poisson solution and cavity
wall velocities. `tests/check_openmp.py` and `tests/check_mpi.py` run small
cases at 1, 2, and 4 workers and compare every saved field with serial.
These are correctness tests; the timing scripts are separate.

`scripts/bench_openmp.py` and `scripts/bench_mpi.py` repeat a fixed case,
reject field differences above `1e-10`, save raw run times, and calculate
median speedup and efficiency. Speedup is `serial median / parallel median`;
efficiency divides that speedup by worker count. Their output goes under
`results/`. `scripts/fields.py` loads CSV fields into NumPy arrays,
`scripts/plot.py` makes the flow plots, and `scripts/validate.py` checks the
grid, wall values, finite numbers, Poisson residual, and interior divergence.
It checks the published Re=100 vortex location only for a run that met the
steady stopping criterion.

## Browser page

`app.py` is a local Flask server. Its `/api/poisson`, `/api/run`,
`/api/openmp`, and `/api/mpi` routes launch the compiled executables with
fixed or validated arguments and return their output. A lock keeps two heavy
runs from starting at once. Results live under `results/web_runs/`.

`web/index.html` contains the controls and result areas. `web/style.css`
handles the layout. `web/app.js` sends requests to Flask and fills the tables
and images with returned results. Numerical work stays in C++; Python makes
plots from its CSV output, and JavaScript displays them.
