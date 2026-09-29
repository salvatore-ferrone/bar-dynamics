#!/usr/bin/env python3
"""Plot quick-look Galactocentric stream projections from an HDF5 run directory."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "simulations" / "pal5_bar_pattern_speed_v1"
DEFAULT_OUTPUT = ROOT / "plots" / "pal5_bar_pattern_speed_v1_quicklook.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    files = [args.input] if args.input.is_file() else sorted(args.input.glob("*.h5"))
    if not files:
        parser.error(f"No HDF5 simulation files found at {args.input}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140, constrained_layout=True)
    runs = []
    control = None
    for path in files:
        with h5py.File(path, "r") as data:
            stream = data["stream/phase_space"][:]
            run_type = data.attrs["run_type"]
            if isinstance(run_type, bytes):
                run_type = run_type.decode()
            if run_type == "axisymmetric_control":
                control = (stream, path.name)
            else:
                speed = float(data.attrs["bar_pattern_speed_magnitude_kms_kpc"])
                runs.append((speed, stream, path.name))

    colors = plt.get_cmap("viridis")
    speed_values = [run[0] for run in runs]
    speed_min = min(speed_values, default=0.0)
    speed_max = max(speed_values, default=1.0)
    normalizer = plt.Normalize(speed_min, speed_max)
    for speed, stream, _ in runs:
        axes[0].scatter(stream[:, 0], stream[:, 1], s=1, alpha=0.18, color=colors(normalizer(speed)))
        axes[1].scatter(stream[:, 0], stream[:, 2], s=1, alpha=0.18, color=colors(normalizer(speed)))

    if control is not None:
        for axis, vertical_index in zip(axes, (1, 2)):
            axis.scatter(
                control[0][:, 0], control[0][:, vertical_index], s=2, color="black", alpha=0.45,
                label="axisymmetric control"
            )
            axis.legend(frameon=False, loc="best")

    axes[0].set(xlabel="Galactocentric x [kpc]", ylabel="y [kpc]", title="In-plane morphology")
    axes[1].set(xlabel="Galactocentric x [kpc]", ylabel="z [kpc]", title="Vertical morphology")
    for axis in axes:
        axis.set_aspect("equal", adjustable="datalim")
    if speed_values:
        scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=colors)
        fig.colorbar(
            scalar_map, ax=axes, location="bottom", pad=0.12, shrink=0.78, aspect=35,
            label="Clockwise bar pattern speed magnitude [km/s/kpc]"
        )
    fig.suptitle(f"Palomar 5: {len(runs)} barred runs" + (" + axisymmetric control" if control else ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"Read {len(files)} simulations; saved {args.output}")


if __name__ == "__main__":
    main()
