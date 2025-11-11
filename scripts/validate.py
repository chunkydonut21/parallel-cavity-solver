"""Check structural and numerical properties of a cavity output CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fields import load_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/cavity.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("results/cavity_metrics.json"))
    args = parser.parse_args()

    x, y, fields = load_grid(args.input)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if len(x) != metrics["grid_size"] or metrics["reynolds"] <= 0:
        raise ValueError("CSV grid and metrics disagree")
    if not np.allclose(np.diff(x), np.diff(x)[0]) or not np.allclose(np.diff(y), np.diff(y)[0]):
        raise ValueError("grid spacing is not uniform")
    if not np.allclose(fields["speed"], np.hypot(fields["u"], fields["v"]), atol=1e-12):
        raise ValueError("speed column disagrees with velocity components")

    u, v, psi = fields["u"], fields["v"], fields["psi"]
    lid = metrics["lid_velocity"]
    if not np.allclose(psi[0, :], 0) or not np.allclose(psi[-1, :], 0) or \
       not np.allclose(psi[:, 0], 0) or not np.allclose(psi[:, -1], 0):
        raise ValueError("streamfunction wall value is not zero")
    if not np.allclose(u[-1, 1:-1], lid) or not np.allclose(v[-1, :], 0):
        raise ValueError("moving lid velocity is incorrect")
    if not np.allclose(u[0, :], 0) or not np.allclose(v[0, :], 0) or \
       not np.allclose(u[:, 0], 0) or not np.allclose(v[:, 0], 0) or \
       not np.allclose(u[:, -1], 0) or not np.allclose(v[:, -1], 0):
        raise ValueError("stationary wall velocity is incorrect")
    if metrics["last_poisson_residual"] > metrics["poisson_tolerance"]:
        raise ValueError("Poisson residual exceeds the configured tolerance")

    # Central differences of the central-difference velocity should commute away from walls.
    h = float(np.diff(x)[0])
    divergence = (u[2:-2, 3:-1] - u[2:-2, 1:-3] +
                  v[3:-1, 2:-2] - v[1:-3, 2:-2]) / (2 * h)
    max_divergence = float(np.max(np.abs(divergence)))
    if max_divergence > 1e-10:
        raise ValueError(f"discrete interior divergence is {max_divergence:g}")

    vortex_j, vortex_i = np.unravel_index(np.argmin(psi), psi.shape)
    vortex_x, vortex_y = float(x[vortex_i]), float(y[vortex_j])
    print(f"grid={len(x)} Re={metrics['reynolds']} steps={metrics['steps']}")
    print(f"max interior discrete divergence={max_divergence:.3e}")
    print(f"primary vortex grid location=({vortex_x:.5f}, {vortex_y:.5f})")
    print(f"steady_converged={metrics['steady_converged']}")
    if metrics["steady_converged"] and abs(metrics["reynolds"] - 100.0) < 1e-12:
        # Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387–411.
        # DOI: 10.1016/0021-9991(82)90058-4. Their Re=100 center is approximate.
        reference = np.array([0.6172, 0.7344])
        distance = float(np.linalg.norm(np.array([vortex_x, vortex_y]) - reference))
        print(f"distance from published Re=100 vortex center={distance:.5f}")
        if distance > 3 * h:
            raise ValueError("primary vortex center is farther than three grid spacings from reference")
    else:
        print("steady reference comparison skipped: flow has not met steady criterion")


if __name__ == "__main__":
    main()
