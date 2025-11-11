# Cavity solver in C++

I built a small lid-driven cavity solver to compare three ways of running the
same numerical work: serial C++, OpenMP, and MPI. The cavity gives the stencil
code a real CFD use case. The part I am studying is the computation: how much
time Jacobi iteration takes, how the grid is split between workers, and what
thread coordination or process communication costs.

The repository includes a local browser page. It runs the C++ executables,
shows the flow field, and displays timings from those runs. It is useful for
trying settings; the command-line tools are also available for repeatable
benchmarks.

## What is here

| Part | What it does |
| --- | --- |
| `cavity_serial` | Serial cavity simulation and the reference output |
| `cavity_openmp` | Same solver code, with selected grid loops split across threads |
| `cavity_mpi` | Row-decomposed solver with ghost-row exchange and global convergence checks |
| `poisson_serial_bench` | Tests the Jacobi solver on a grid problem with a known answer |
| `app.py` and `web/` | Local page for runs, plots, and comparisons |
| `scripts/` | Plotting, validation, and repeated benchmark runs |

The implementation is in `src/`. [Code notes](docs/code-walkthrough.md) explain
the data layout and the parallel code. [Benchmarks](docs/benchmarks.md) contain
the measurements so far and the commands used to produce them.

## Build on Windows

Install Visual Studio Build Tools with **Desktop development with C++** and
CMake. For the MPI executable, install the Microsoft MPI runtime and SDK from
[Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=105289).
I used these package-manager commands:

```powershell
winget install --id Microsoft.msmpi --exact
winget install --id Microsoft.msmpisdk --exact
```

From this directory, build the Release configuration and run the tests:

```powershell
$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = $cmake.Replace("cmake.exe", "ctest.exe")
& $cmake -S . -B build -G "Visual Studio 18 2026"
& $cmake --build build --config Release
& $ctest --test-dir build -C Release --output-on-failure
```

The CMake and CTest commands can be shortened to `cmake` and `ctest` if they
are already on your `PATH`. CMake uses the system MPI installation on Linux.

## Open the local page

The page needs Python for Flask and plotting. Use a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:8765>. Stop the server with Ctrl+C. After setup, only
the last command is needed. The page runs a Poisson benchmark, a cavity case,
and fixed-work OpenMP and MPI comparisons. It does not use stored benchmark
numbers. Results go under `results/web_runs/`.

## Run from the command line

The serial solver writes a CSV field and a JSON file with run settings and
timings. These commands run a 65 × 65 case, plot the fields, and check the
output:

```powershell
.\build\Release\cavity_serial.exe --grid 65 --re 100 --steps 4000
.\.venv\Scripts\python.exe scripts\plot.py
.\.venv\Scripts\python.exe scripts\validate.py
```

To run four MPI processes, give that run its own output paths:

```powershell
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 .\build\Release\cavity_mpi.exe --grid 65 --steps 100 --output results\mpi_4.csv --metrics results\mpi_4.json
.\.venv\Scripts\python.exe scripts\validate.py --input results\mpi_4.csv --metrics results\mpi_4.json
```

There must be at least one interior row per MPI process, so the process count
cannot exceed `N-2`. Only rank 0 writes the final files.

For a steady-flow check, run longer and enable the stopping criterion:

```powershell
.\build\Release\cavity_serial.exe --grid 65 --steps 20000 --steady-tol 0.001 --min-steps 4000 --output results\cavity_steady.csv --metrics results\cavity_steady_metrics.json
.\.venv\Scripts\python.exe scripts\validate.py --input results\cavity_steady.csv --metrics results\cavity_steady_metrics.json
```

Short runs on the page show the flow while it is still changing. The plots
alone cannot establish a steady solution.

## How the solver works

The top wall moves right; the other walls are stationary. The code stores each
`N × N` field in a flat row-major `std::vector<double>`. It solves the
vorticity–streamfunction equations on that grid:

```text
∇²ψ = −ω
u = ∂ψ/∂y,  v = −∂ψ/∂x
∂ω/∂t + u ∂ω/∂x + v ∂ω/∂y = ν ∇²ω
```

The Poisson solve uses a five-point stencil and Jacobi iteration. Velocity
comes from central differences. Vorticity moves with an explicit Euler step,
upwind differences for advection, and a centered diffusion stencil. Separate
old and new grids make the updates independent. The default timestep is a
conservative fixed value chosen from grid spacing, lid speed, and viscosity;
`--dt` overrides it.

Jacobi is easy to partition, but it needs many sweeps as the grid grows.
First-order upwind transport also adds numerical diffusion. These are known
limits of this implementation. The Poisson benchmark checks a manufactured
discrete solution; the cavity validator checks boundary values, divergence,
and the residual. A steady Re=100 run also checks the primary vortex location
against [Ghia, Ghia & Shin (1982)](https://doi.org/10.1016/0021-9991(82)90058-4).

## What the timings mean

The reported numerical runtime includes the solver steps and, for MPI, row
exchange and reductions. It excludes process startup, plotting, CSV/JSON
writing, and the final MPI gather for output. The OpenMP and MPI comparison
scripts run each setup three times, check saved fields against serial, and
report the median. Their default case is 65 × 65 for 100 timesteps, which is
a short fixed-work test. See [benchmarks](docs/benchmarks.md) for the results
and their limits.

`build/`, `.venv/`, and `results/` are generated locally and ignored by Git.
