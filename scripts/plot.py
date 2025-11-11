"""Plot a cavity_serial CSV without changing its numerical data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from fields import load_grid


def plot_velocity(x: np.ndarray, y: np.ndarray, fields: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    contour = ax.contourf(x, y, fields["speed"], levels=30, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Velocity magnitude")
    ax.streamplot(x, y, fields["u"], fields["v"], color="white", density=1.2,
                  linewidth=0.7, arrowsize=0.8)
    ax.set(xlabel="x", ylabel="y", title="Lid-driven cavity: velocity and streamlines")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_vorticity(x: np.ndarray, y: np.ndarray, fields: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    bound = max(float(np.max(np.abs(fields["omega"]))), 1.0)
    positive_levels = np.geomspace(0.1, bound, 16)
    levels = np.concatenate((-positive_levels[::-1], [0.0], positive_levels))
    contour = ax.contourf(x, y, fields["omega"], levels=levels, cmap="RdBu_r",
                          norm=SymLogNorm(linthresh=0.1, vmin=-bound, vmax=bound))
    fig.colorbar(contour, ax=ax, label="Vorticity")
    ax.set(xlabel="x", ylabel="y", title="Lid-driven cavity: vorticity")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/cavity.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    x, y, fields = load_grid(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_velocity(x, y, fields, args.output_dir / "velocity_streamlines.png")
    plot_vorticity(x, y, fields, args.output_dir / "vorticity.png")
    print(f"Saved plots in {args.output_dir}")


if __name__ == "__main__":
    main()
