#!/usr/bin/env python3
"""Plot a quick-look overlay of all pattern speeds from a merged experiment HDF5 file."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "plots" / "pal5_bar_pattern_speed_v1_quicklook.png"


def _text_attribute(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def find_default_simulation_file(stream_id="pal5"):
    """Locate the most recently written merged experiment file for this stream."""
    candidates = sorted(
        (ROOT / "simulations").glob(f"{stream_id}_bar_pattern_speed_*.h5"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No merged experiment HDF5 files found in {ROOT / 'simulations'}")
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    simulation_file = args.input or find_default_simulation_file()
    if not simulation_file.is_file():
        parser.error(f"Simulation file does not exist: {simulation_file}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140, constrained_layout=True)
    runs = []
    control = None
    with h5py.File(simulation_file, "r") as simulation:
        for run_key, run_group in simulation["runs"].items():
            stream = run_group["stream_phase_space"][:]
            run_type = _text_attribute(run_group.attrs.get("run_type", ""))
            if run_type == "axisymmetric_control":
                control = (stream, run_key)
            else:
                speed = float(run_group.attrs["bar_pattern_speed_magnitude_kms_kpc"])
                runs.append((speed, stream, run_key))

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
    print(f"Read {len(runs) + (1 if control else 0)} runs from {simulation_file}; saved {args.output}")


if __name__ == "__main__":
    main()

