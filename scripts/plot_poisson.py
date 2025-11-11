"""Plot measured serial Poisson cost across grid sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="poisson_serial_bench CSV files")
    parser.add_argument("--output", type=Path, default=Path("results/poisson_scaling.png"))
    args = parser.parse_args()
    frame = pd.concat((pd.read_csv(path) for path in args.inputs), ignore_index=True)
    if frame["tolerance"].nunique() != 1 or frame["implementation"].nunique() != 1:
        raise ValueError("plot only one implementation and residual tolerance at a time")
    summary = frame.groupby("grid_size", as_index=False).agg(
        median_seconds=("seconds", "median"),
        median_iterations=("iterations", "median"),
    ).sort_values("grid_size")
    n = summary["grid_size"].to_numpy(dtype=float)
    seconds = summary["median_seconds"].to_numpy(dtype=float)
    iterations = summary["median_iterations"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].loglog(n, iterations, "o-", color="#155e75", label="measured")
    axes[0].set(xlabel="Grid points per side", ylabel="Jacobi sweeps",
                title="Iterations to common residual")
    axes[1].loglog(n, seconds, "o-", color="#9a3412", label="measured median")
    if len(n) > 1:
        guide = seconds[0] * ((n - 1) / (n[0] - 1)) ** 4
        axes[1].loglog(n, guide, "--", color="0.4", label="N⁴ guide")
    axes[1].set(xlabel="Grid points per side", ylabel="Seconds",
                title="Time to common residual")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
