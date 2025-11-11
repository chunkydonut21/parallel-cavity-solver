"""Local browser interface for running and inspecting the C++ solver."""

import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "web_runs"
SOLVER = ROOT / "build" / "Release" / "cavity_serial.exe"
POISSON_BENCH = ROOT / "build" / "Release" / "poisson_serial_bench.exe"
OPENMP_SOLVER = ROOT / "build" / "Release" / "cavity_openmp.exe"
MPI_SOLVER = ROOT / "build" / "Release" / "cavity_mpi.exe"
PLOT_SCRIPT = ROOT / "scripts" / "plot.py"
OPENMP_BENCH_SCRIPT = ROOT / "scripts" / "bench_openmp.py"
MPI_BENCH_SCRIPT = ROOT / "scripts" / "bench_mpi.py"
RUN_ID = re.compile(r"[0-9a-f]{12}")

app = Flask(__name__, static_folder="web", static_url_path="")
run_lock = threading.Lock()


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/api/run")
def run_solver():
    data = request.get_json(silent=True) or {}
    grid = data.get("grid")
    reynolds = data.get("reynolds")
    steps = data.get("steps")

    if type(grid) is not int or grid not in (17, 33, 65):
        return jsonify(error="Choose a grid of 17, 33, or 65 points per side."), 400
    if type(steps) is not int or not 100 <= steps <= 4000:
        return jsonify(error="Steps must be between 100 and 4000."), 400
    if type(reynolds) not in (int, float) or not math.isfinite(reynolds) or not 25 <= reynolds <= 400:
        return jsonify(error="Reynolds number must be between 25 and 400."), 400
    if not SOLVER.is_file():
        return jsonify(error="Release solver not found. Build cavity_serial in Release mode first."), 503
    if not run_lock.acquire(blocking=False):
        return jsonify(error="A simulation is already running. Try again when it finishes."), 429

    try:
        run_id = uuid.uuid4().hex[:12]
        folder = RUNS / run_id
        folder.mkdir(parents=True)
        csv_path = folder / "cavity.csv"
        metrics_path = folder / "metrics.json"
        command = [
            str(SOLVER), "--grid", str(grid), "--re", str(reynolds),
            "--steps", str(steps), "--output", str(csv_path),
            "--metrics", str(metrics_path),
        ]
        subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                       check=True, timeout=240)
        plot_env = os.environ.copy()
        plot_env["MPLCONFIGDIR"] = str(folder / ".mplconfig")
        subprocess.run([sys.executable, str(PLOT_SCRIPT), "--input", str(csv_path),
                        "--output-dir", str(folder)], cwd=ROOT, env=plot_env,
                       capture_output=True, text=True, check=True, timeout=30)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["run_id"] = run_id
        metrics["velocity_image"] = f"/runs/{run_id}/velocity_streamlines.png"
        metrics["vorticity_image"] = f"/runs/{run_id}/vorticity.png"
        return jsonify(metrics)
    except subprocess.TimeoutExpired:
        return jsonify(error="The run exceeded the four-minute local demo limit."), 504
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "Solver failed").strip()
        return jsonify(error=detail[:400]), 500
    finally:
        run_lock.release()


@app.post("/api/poisson")
def benchmark_poisson():
    """Measure the same manufactured Poisson problem at four grid sizes."""
    if not POISSON_BENCH.is_file():
        return jsonify(error="Release Poisson benchmark not found. Build the project in Release mode first."), 503
    if not run_lock.acquire(blocking=False):
        return jsonify(error="A calculation is already running. Try again when it finishes."), 429

    try:
        run_id = uuid.uuid4().hex[:12]
        folder = RUNS / run_id
        folder.mkdir(parents=True)
        measurements = []

        batches = {17: 200, 33: 20, 65: 3, 129: 1}
        for grid, batch in batches.items():
            csv_path = folder / f"poisson_{grid}.csv"
            command = [
                str(POISSON_BENCH), "--grid", str(grid), "--tol", "1e-6",
                "--max-iters", "80000", "--repeats", "3", "--batch", str(batch),
                "--output", str(csv_path),
            ]
            subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                           check=True, timeout=90)
            with csv_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            if len(rows) != 3:
                raise ValueError("Poisson benchmark did not write three measurements")

            measurements.append({
                "grid_size": grid,
                "interior_cells": (grid - 2) ** 2,
                "solves_per_batch": batch,
                "iterations": int(rows[0]["iterations"]),
                "residual": float(rows[0]["residual"]),
                "max_error": float(rows[0]["max_error"]),
                "median_seconds": statistics.median(float(row["seconds"]) for row in rows),
            })

        return jsonify(run_id=run_id, tolerance=1e-6, repeats=3, results=measurements)
    except subprocess.TimeoutExpired:
        return jsonify(error="The Poisson benchmark exceeded its local time limit."), 504
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "Benchmark failed").strip()
        return jsonify(error=detail[:400]), 500
    except (OSError, ValueError, KeyError) as error:
        return jsonify(error=str(error)), 500
    finally:
        run_lock.release()


@app.post("/api/openmp")
def benchmark_openmp():
    """Repeat a fixed-size serial/OpenMP comparison with field checks."""
    if not SOLVER.is_file() or not OPENMP_SOLVER.is_file():
        return jsonify(error="Build cavity_serial and cavity_openmp in Release mode first."), 503
    if not run_lock.acquire(blocking=False):
        return jsonify(error="A calculation is already running. Try again when it finishes."), 429

    try:
        run_id = uuid.uuid4().hex[:12]
        folder = RUNS / run_id
        folder.mkdir(parents=True)
        command = [
            sys.executable, str(OPENMP_BENCH_SCRIPT),
            "--grid", "65", "--steps", "100", "--repeats", "3",
            "--threads", "1", "2", "4", "--output-dir", str(folder),
        ]
        subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                       check=True, timeout=180)

        with (folder / "summary.csv").open(newline="", encoding="utf-8") as file:
            summary_rows = list(csv.DictReader(file))
        with (folder / "runs.csv").open(newline="", encoding="utf-8") as file:
            run_rows = list(csv.DictReader(file))
        summary = [{
            "implementation": row["implementation"],
            "threads": int(row["threads"]),
            "median_seconds": float(row["median_seconds"]),
            "speedup": float(row["speedup"]),
            "efficiency": float(row["efficiency"]),
        } for row in summary_rows]
        max_difference = max(float(row["max_field_difference"]) for row in run_rows)
        return jsonify(run_id=run_id, grid_size=65, steps=100, repeats=3,
                       max_field_difference=max_difference, results=summary)
    except subprocess.TimeoutExpired:
        return jsonify(error="The OpenMP comparison exceeded its local time limit."), 504
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "Benchmark failed").strip()
        return jsonify(error=detail[:400]), 500
    except (OSError, ValueError, KeyError) as error:
        return jsonify(error=str(error)), 500
    finally:
        run_lock.release()


@app.post("/api/mpi")
def benchmark_mpi():
    """Repeat a fixed-size serial/MPI comparison with field checks."""
    if not SOLVER.is_file() or not MPI_SOLVER.is_file():
        return jsonify(error="Build cavity_serial and cavity_mpi in Release mode first."), 503
    if not run_lock.acquire(blocking=False):
        return jsonify(error="A calculation is already running. Try again when it finishes."), 429

    try:
        run_id = uuid.uuid4().hex[:12]
        folder = RUNS / run_id
        folder.mkdir(parents=True)
        command = [
            sys.executable, str(MPI_BENCH_SCRIPT),
            "--grid", "65", "--steps", "100", "--repeats", "3",
            "--ranks", "1", "2", "4", "--output-dir", str(folder),
        ]
        subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                       check=True, timeout=240)
        with (folder / "summary.csv").open(newline="", encoding="utf-8") as file:
            summary_rows = list(csv.DictReader(file))
        with (folder / "runs.csv").open(newline="", encoding="utf-8") as file:
            run_rows = list(csv.DictReader(file))
        summary = [{
            "implementation": row["implementation"],
            "ranks": int(row["ranks"]),
            "median_seconds": float(row["median_seconds"]),
            "speedup": float(row["speedup"]),
            "efficiency": float(row["efficiency"]),
        } for row in summary_rows]
        max_difference = max(float(row["max_field_difference"]) for row in run_rows)
        return jsonify(run_id=run_id, grid_size=65, steps=100, repeats=3,
                       max_field_difference=max_difference, results=summary)
    except subprocess.TimeoutExpired:
        return jsonify(error="The MPI comparison exceeded its local time limit."), 504
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "Benchmark failed").strip()
        return jsonify(error=detail[:400]), 500
    except (OSError, ValueError, KeyError) as error:
        return jsonify(error=str(error)), 500
    finally:
        run_lock.release()


@app.get("/runs/<run_id>/<filename>")
def run_image(run_id, filename):
    if not RUN_ID.fullmatch(run_id) or filename not in ("velocity_streamlines.png", "vorticity.png"):
        return jsonify(error="Image not found."), 404
    return send_from_directory(RUNS / run_id, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False)
