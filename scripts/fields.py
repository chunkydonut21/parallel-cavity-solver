"""Load a solver CSV into square, row-major NumPy fields."""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"x", "y", "u", "v", "speed", "omega", "psi"}


def load_grid(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing CSV columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("CSV contains missing values")
    x = np.sort(frame["x"].unique())
    y = np.sort(frame["y"].unique())
    if len(x) != len(y) or len(frame) != len(x) * len(y):
        raise ValueError("expected one complete square Cartesian grid")
    fields = {}
    for name in ("u", "v", "speed", "omega", "psi"):
        table = frame.pivot(index="y", columns="x", values=name).reindex(index=y, columns=x)
        values = table.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains missing or non-finite values")
        fields[name] = values
    return x, y, fields
