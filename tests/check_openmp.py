"""Run the same cavity case with serial and OpenMP, then compare fields."""

import csv
import json
import subprocess
import sys
from pathlib import Path


def run(executable, output_dir, name, threads=None):
    csv_path = output_dir / f"{name}.csv"
    metrics_path = output_dir / f"{name}.json"
    command = [str(executable)]
    if threads is not None:
        command += ["--threads", str(threads)]
    command += [
        "--grid", "17", "--re", "100", "--steps", "100",
        "--output", str(csv_path), "--metrics", str(metrics_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with csv_path.open(newline="", encoding="utf-8") as file:
        fields = list(csv.DictReader(file))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return fields, metrics


def main():
    serial_exe = Path(sys.argv[1])
    openmp_exe = Path(sys.argv[2])
    output_dir = Path(sys.argv[3]) / "openmp_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference, serial_metrics = run(serial_exe, output_dir, "serial")
    assert serial_metrics["implementation"] == "serial"

    for threads in (1, 2, 4):
        actual, metrics = run(openmp_exe, output_dir, f"openmp_{threads}", threads)
        assert metrics["implementation"] == "openmp"
        assert metrics["threads"] == threads
        assert len(actual) == len(reference)

        largest_error = 0.0
        for expected_row, actual_row in zip(reference, actual):
            for field in ("x", "y", "u", "v", "speed", "omega", "psi"):
                difference = abs(float(expected_row[field]) - float(actual_row[field]))
                largest_error = max(largest_error, difference)

        print(f"threads={threads} maximum field difference={largest_error:.3g}")
        assert largest_error < 1e-10
        assert abs(metrics["last_poisson_residual"] - serial_metrics["last_poisson_residual"]) < 1e-10


if __name__ == "__main__":
    main()
