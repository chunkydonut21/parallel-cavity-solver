"""Check complete MPI fields against the serial cavity result."""

import csv
import json
import subprocess
import sys
from pathlib import Path


def run(command, output_dir, name):
    csv_path = output_dir / f"{name}.csv"
    metrics_path = output_dir / f"{name}.json"
    command += [
        "--grid", "17", "--re", "100", "--steps", "100",
        "--output", str(csv_path), "--metrics", str(metrics_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    with csv_path.open(newline="", encoding="utf-8") as file:
        fields = list(csv.DictReader(file))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return fields, metrics


def main():
    mpiexec = Path(sys.argv[1])
    serial_exe = Path(sys.argv[2])
    mpi_exe = Path(sys.argv[3])
    output_dir = Path(sys.argv[4]) / "mpi_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference, serial_metrics = run([str(serial_exe)], output_dir, "serial")
    for ranks in (1, 2, 4):
        actual, metrics = run([str(mpiexec), "-n", str(ranks), str(mpi_exe)],
                              output_dir, f"mpi_{ranks}")
        if len(actual) != len(reference) or metrics["ranks"] != ranks:
            raise AssertionError(f"wrong output size or rank count for {ranks} ranks")

        largest_error = 0.0
        for expected_row, actual_row in zip(reference, actual):
            for field in ("x", "y", "u", "v", "speed", "omega", "psi"):
                difference = abs(float(expected_row[field]) - float(actual_row[field]))
                largest_error = max(largest_error, difference)
        print(f"ranks={ranks} maximum field difference={largest_error:.3g}")
        if largest_error > 1e-10:
            raise AssertionError(f"MPI result differs from serial by {largest_error}")
        if abs(metrics["last_poisson_residual"] - serial_metrics["last_poisson_residual"]) > 1e-10:
            raise AssertionError("MPI and serial Poisson residuals differ")


if __name__ == "__main__":
    main()
