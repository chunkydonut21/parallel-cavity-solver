"""Measure a fixed cavity workload with serial and MPI ranks."""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""


def read_fields(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def max_difference(reference, actual):
    if len(reference) != len(actual):
        raise ValueError("field files have different row counts")
    largest = 0.0
    for left, right in zip(reference, actual):
        for name in ("x", "y", "u", "v", "speed", "omega", "psi"):
            largest = max(largest, abs(float(left[name]) - float(right[name])))
    return largest


def run_case(command, grid, reynolds, steps, output_dir, label):
    field_path = output_dir / f"{label}.csv"
    metrics_path = output_dir / f"{label}.json"
    command += [
        "--grid", str(grid), "--re", str(reynolds), "--steps", str(steps),
        "--output", str(field_path), "--metrics", str(metrics_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True,
                   text=True, timeout=180)
    return (json.loads(metrics_path.read_text(encoding="utf-8")),
            read_fields(field_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=65)
    parser.add_argument("--re", type=float, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mpi_bench")
    args = parser.parse_args()
    if args.grid < 5 or args.re <= 0 or args.steps < 1 or args.repeats < 1:
        parser.error("grid, Reynolds number, steps, and repeats must be positive")
    if any(ranks < 1 or ranks > args.grid - 2 for ranks in args.ranks):
        parser.error("ranks must be between 1 and the number of interior rows")

    serial = ROOT / "build" / "Release" / f"cavity_serial{EXE}"
    mpi = ROOT / "build" / "Release" / f"cavity_mpi{EXE}"
    if not serial.is_file() or not mpi.is_file():
        parser.error("build cavity_serial and cavity_mpi in Release mode first")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None and os.name == "nt":
        candidate = Path("C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")
        if candidate.is_file():
            mpiexec = str(candidate)
    if mpiexec is None:
        parser.error("mpiexec was not found")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = []
    reference = None
    cases = [("serial", 1)] + [("mpi", ranks) for ranks in args.ranks]
    for implementation, ranks in cases:
        for repeat in range(1, args.repeats + 1):
            label = f"{implementation}_{ranks}_{repeat}"
            command = ([str(serial)] if implementation == "serial"
                       else [mpiexec, "-n", str(ranks), str(mpi)])
            metrics, fields = run_case(command, args.grid, args.re, args.steps,
                                       output_dir, label)
            if reference is None:
                reference = fields
            difference = max_difference(reference, fields)
            if difference > 1e-10:
                raise ValueError(f"{label} differs from serial by {difference}")
            raw.append({
                "implementation": implementation, "ranks": ranks, "repeat": repeat,
                "grid_size": args.grid, "reynolds": args.re, "steps": args.steps,
                "seconds": metrics["runtime_seconds"],
                "poisson_seconds": metrics["poisson_seconds"],
                "max_field_difference": difference,
            })
            print(f"{label}: {metrics['runtime_seconds']:.4f} s, max field difference {difference:.3g}")

    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)

    serial_median = statistics.median(row["seconds"] for row in raw
                                      if row["implementation"] == "serial")
    summary = []
    for implementation, ranks in cases:
        times = [row["seconds"] for row in raw
                 if row["implementation"] == implementation and row["ranks"] == ranks]
        median = statistics.median(times)
        speedup = serial_median / median
        summary.append({
            "implementation": implementation, "ranks": ranks,
            "median_seconds": median, "speedup": speedup,
            "efficiency": speedup / ranks,
        })
        print(f"{implementation} ranks={ranks}: median {median:.4f} s, speedup {speedup:.3f}")

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
