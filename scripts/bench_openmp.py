"""Measure serial and OpenMP cavity runs at one fixed problem size."""

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""


def read_fields(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def max_field_difference(reference, actual):
    if len(reference) != len(actual):
        raise ValueError("field files have different row counts")
    difference = 0.0
    for left, right in zip(reference, actual):
        for name in ("x", "y", "u", "v", "speed", "omega", "psi"):
            difference = max(difference, abs(float(left[name]) - float(right[name])))
    return difference


def run_case(executable, grid, reynolds, steps, threads, output_dir, label):
    field_path = output_dir / f"{label}.csv"
    metrics_path = output_dir / f"{label}.json"
    command = [str(executable)]
    if threads is not None:
        command += ["--threads", str(threads)]
    command += [
        "--grid", str(grid), "--re", str(reynolds), "--steps", str(steps),
        "--output", str(field_path), "--metrics", str(metrics_path),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(metrics_path.read_text(encoding="utf-8")), read_fields(field_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=65)
    parser.add_argument("--re", type=float, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "openmp_bench")
    args = parser.parse_args()
    if args.grid < 5 or args.re <= 0 or args.steps < 1 or args.repeats < 1:
        parser.error("grid, Reynolds number, steps, and repeats must be positive")
    if any(threads < 1 for threads in args.threads):
        parser.error("thread counts must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    serial = ROOT / "build" / "Release" / f"cavity_serial{EXE}"
    openmp = ROOT / "build" / "Release" / f"cavity_openmp{EXE}"
    if not serial.is_file() or not openmp.is_file():
        parser.error("build cavity_serial and cavity_openmp in Release mode first")

    raw = []
    reference = None
    for implementation, threads in [("serial", 1)] + [("openmp", n) for n in args.threads]:
        for repeat in range(1, args.repeats + 1):
            executable = serial if implementation == "serial" else openmp
            label = f"{implementation}_{threads}_{repeat}"
            metrics, fields = run_case(executable, args.grid, args.re, args.steps,
                                       None if implementation == "serial" else threads,
                                       output_dir, label)
            if reference is None:
                reference = fields
            difference = max_field_difference(reference, fields)
            if difference > 1e-10:
                raise ValueError(f"{label} differs from serial by {difference}")
            raw.append({
                "implementation": implementation, "threads": threads, "repeat": repeat,
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

    serial_median = statistics.median(row["seconds"] for row in raw if row["implementation"] == "serial")
    summary = []
    for implementation, threads in [("serial", 1)] + [("openmp", n) for n in args.threads]:
        times = [row["seconds"] for row in raw
                 if row["implementation"] == implementation and row["threads"] == threads]
        median = statistics.median(times)
        speedup = serial_median / median
        summary.append({
            "implementation": implementation, "threads": threads,
            "median_seconds": median, "speedup": speedup,
            "efficiency": speedup / threads,
        })
        print(f"{implementation} threads={threads}: median {median:.4f} s, speedup {speedup:.3f}")

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
