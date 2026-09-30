#!/usr/bin/env python3
"""Save one Galactocentric y-z figure for each barred pattern-speed run."""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import plot_types


ROOT = Path(__file__).resolve().parents[1]

GENERAL_AXIS = {"aspect": "equal"}
SPEED_MIN, SPEED_MAX = 25.0, 60.0
CMAP = plt.get_cmap("rainbow")
NORM = Normalize(vmin=SPEED_MIN, vmax=SPEED_MAX)

# Per-stream presentation settings for this plot type (Galactocentric y-z).
PAL5 = {
    "stream_id": "pal5",
    "x_column": 1,
    "y_column": 2,
    "xlim": (-10.0, 10.0),
    "ylim": (-10.0, 20.0),
    "axis": {
        "xlabel": r"Galactocentric $y$ [$\rm{kpc}$]",
        "ylabel": r"Galactocentric $z$ [$\rm{kpc}$]",
    },
}

NGC4590 = {
    "stream_id": "ngc4590",
    "x_column": 1,
    "y_column": 2,
    "xlim": (-24.0, 22.0),
    "ylim": (-16.0, 10.0),
    "axis": {
        "xlabel": r"Galactocentric $y$ [$\rm{kpc}$]",
        "ylabel": r"Galactocentric $z$ [$\rm{kpc}$]",
    },
}

NGC3201 = {
    "stream_id": "ngc3201",
    "x_column": 1,
    "y_column": 2,
    "xlim": None,
    "ylim": None,
    "axis": {
        "xlabel": r"Galactocentric $y$ [$\rm{kpc}$]",
        "ylabel": r"Galactocentric $z$ [$\rm{kpc}$]",
    },
}

STREAMS = {"pal5": PAL5, "ngc4590": NGC4590, "ngc3201": NGC3201}


def _text_attribute(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def find_default_simulation_file(stream_id):
    """Locate the most recently written merged experiment file for this stream."""
    candidates = [
        path for path in sorted(
            (ROOT / "simulations").glob("*.h5"), key=lambda path: path.stat().st_mtime
        )
        if _file_stream_id(path) == stream_id
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No merged experiment HDF5 files with stream_id={stream_id!r} found in {ROOT / 'simulations'}"
        )
    return candidates[-1]


def _file_stream_id(path):
    with h5py.File(path, "r") as simulation:
        return _text_attribute(simulation.attrs.get("stream_id", ""))


def _padded_limits(values):
    low = float(np.min(values))
    high = float(np.max(values))
    padding = max((high - low) * 0.05, 0.5)
    return low - padding, high + padding


def load_barred_runs(simulation_file, stream_id):
    """Load matching barred runs as (run_key, speed, phase_space) tuples."""
    runs = []
    with h5py.File(simulation_file, "r") as simulation:
        if _text_attribute(simulation.attrs.get("stream_id", "")) != stream_id:
            raise ValueError(f"{simulation_file} does not hold stream_id={stream_id!r}")
        for run_key, run_group in simulation["runs"].items():
            if _text_attribute(run_group.attrs.get("run_type", "")) != "barred":
                continue
            speed = float(run_group.attrs["bar_pattern_speed_magnitude_kms_kpc"])
            phase_space = run_group["stream_phase_space"][:]
            runs.append((run_key, speed, phase_space))

    return sorted(runs, key=lambda run: run[1])


def plot_run(run_key, speed, phase_space, settings, output_dir):
    """Render and save one simulation using the selected stream's presentation."""
    fig, axis, color_axis = plot_types.flush_color_bar_single_column(
        settings["xlim"], settings["ylim"]
    )
    axis.scatter(
        phase_space[:, settings["x_column"]],
        phase_space[:, settings["y_column"]],
        s=3,
        color=CMAP(NORM(speed)),
        linewidths=0,
    )
    axis.set(
        xlim=settings["xlim"],
        ylim=settings["ylim"],
        **GENERAL_AXIS,
        **settings["axis"],
    )

    colorbar = fig.colorbar(
        ScalarMappable(norm=NORM, cmap=CMAP),
        cax=color_axis,
        label=r"$|\Omega_\mathrm{bar}|$ [km s$^{-1}$ kpc$^{-1}$]",
    )
    colorbar.ax.axhline(speed, color="black", linewidth=1.5, zorder=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_key}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def sweep_pattern_speeds(simulation_file, output_dir, stream_settings, limit=None):
    """Make one fixed-scale plot per barred run for the requested stream."""
    runs = load_barred_runs(simulation_file, stream_settings["stream_id"])
    if limit is not None:
        runs = runs[:limit]
    if not runs:
        raise FileNotFoundError(
            f"No barred runs for stream {stream_settings['stream_id']!r} "
            f"were found in {simulation_file}"
        )

    if stream_settings["xlim"] is None or stream_settings["ylim"] is None:
        coordinates = np.concatenate(
            [run[2][:, [stream_settings["x_column"], stream_settings["y_column"]]] for run in runs]
        )
        stream_settings = {
            **stream_settings,
            "xlim": _padded_limits(coordinates[:, 0]),
            "ylim": _padded_limits(coordinates[:, 1]),
        }

    output_paths = []
    for run_key, speed, phase_space in runs:
        output_path = plot_run(run_key, speed, phase_space, stream_settings, output_dir)
        output_paths.append(output_path)
        print(f"{speed:5.1f} km/s/kpc -> {output_path}")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=sorted(STREAMS), default=None)
    parser.add_argument("--simulation-file", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, help="only render the first N speeds (for testing)")
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="render text with LaTeX and the txfonts package (requires a TeX installation)",
    )
    args = parser.parse_args()

    if not args.usetex:
        plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "stix"})

    if args.simulation_file is not None:
        simulation_file = args.simulation_file
        if not simulation_file.is_file():
            parser.error(f"Simulation file does not exist: {simulation_file}")
        stream_id = args.stream or _file_stream_id(simulation_file)
    else:
        stream_id = args.stream or "pal5"
        simulation_file = find_default_simulation_file(stream_id)
    if stream_id not in STREAMS:
        parser.error(f"No plotting settings are defined for stream_id={stream_id!r}")
    stream_settings = STREAMS[stream_id]
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    plot_dir = args.plot_dir or ROOT / "plots" / f"{stream_id}_bar_pattern_speed_v1"
    outputs = sweep_pattern_speeds(
        simulation_file,
        plot_dir,
        stream_settings,
        limit=args.limit,
    )
    print(f"Saved {len(outputs)} figures to {plot_dir}")


if __name__ == "__main__":
    main()

