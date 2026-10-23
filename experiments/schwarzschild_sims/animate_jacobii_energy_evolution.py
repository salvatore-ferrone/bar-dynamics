"""Render Jacobi-energy evolution frames in parallel.

usage: python animate_jacobii_energy_evolution.py ferrers_bar_10000.h5 nbody_10000
then:  ffmpeg -framerate 60 -i ferrers_bar_10000/nbody_10000/frame_%06d.png \
           -pix_fmt yuv420p jacobii_energy.mp4
"""
import argparse
import atexit
import os
import pathlib
from multiprocessing import Pool

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy
from astropy import units as u

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{txfonts}",
})

_W = {}


def read_jacobii(worker, indices):
    """Read and compute the Jacobi energies for selected timesteps."""
    mass = worker["mass"]
    trajectories = worker["trajectories"][indices, :, :]
    potential_energy_external = mass * worker["potential_energy_external"][indices, :]
    potential_energy_internal = 0.5 * mass * worker["potential_energy_internal"][indices, :]
    kinetic_energy = 0.5 * mass * numpy.sum(trajectories[:, :, 3:6] ** 2, axis=2)
    angular_momentum_z = (
        trajectories[:, :, 0] * trajectories[:, :, 4]
        - trajectories[:, :, 1] * trajectories[:, :, 3]
    )
    return (
        potential_energy_external
        + potential_energy_internal
        + kinetic_energy
        - worker["omega"] * angular_momentum_z
    )


def init_worker(path, group, outdir, bins, norm_min, norm_max, unit_energy,
                history_stride, dpi):
    file = h5py.File(path, "r")
    data = file[group]
    mass = data["mass"][:]
    times = (
        data["time"][:] * data.attrs["unitT"] * u.s * u.kpc / u.km
    ).to(u.Gyr).value
    omega = file["schwarzschild"].attrs["omega"]

    norm = matplotlib.colors.Normalize(vmin=norm_min, vmax=norm_max)
    cmap = plt.get_cmap("rainbow")
    initial_jacobii = read_jacobii({
        "mass": mass,
        "trajectories": data["trajectories"],
        "potential_energy_external": data["potential_energy_external"],
        "potential_energy_internal": data["potential_energy_internal"],
        "omega": omega,
    }, [0])[0]
    colors = cmap(norm(initial_jacobii))

    fig, axes = plt.subplots(1, 2, figsize=(6.26, 3.13))
    histogram_axis, evolution_axis = axes
    histogram_axis.set(
        xlabel=r"$E_j~[10^3~\mathrm{km}^2/\mathrm{s}^2]$",
        ylabel=r"$\mathrm{d}n/\mathrm{d}E_j$",
        xlim=[bins.min(), bins.max()],
        ylim=[0, 200],
        title="Ferrer's potential",
    )
    histogram_axis.text(
        0.99, 0.99, "N: {:d}".format(mass.shape[0]),
        transform=histogram_axis.transAxes, ha="right", va="top",
    )
    _, _, bars = histogram_axis.hist([], bins=bins, edgecolor="black")
    centers = (bins[:-1] + bins[1:]) / 2
    for bar, color in zip(bars, cmap(norm(centers / unit_energy))):
        bar.set_facecolor(color)
        bar.set_edgecolor(color)

    evolution_axis.set(
        xlim=[0, times.max()],
        ylim=[unit_energy * norm_min * 2, unit_energy * norm_max * 2],
        ylabel=r"$E_j~[10^3~\mathrm{km}^2/\mathrm{s}^2]$",
        xlabel="Time [Gyr]",
    )
    line_collection = LineCollection([], colors=colors, alpha=0.1)
    evolution_axis.add_collection(line_collection)
    current_points = evolution_axis.scatter(
        numpy.full(mass.shape[0], numpy.nan),
        numpy.full(mass.shape[0], numpy.nan),
        color=colors,
        s=5,
        zorder=1000,
    )
    time_label = histogram_axis.text(
        0.01, 0.99, "", transform=histogram_axis.transAxes, ha="left", va="top"
    )

    worker = {
        "file": file,
        "mass": mass,
        "trajectories": data["trajectories"],
        "potential_energy_external": data["potential_energy_external"],
        "potential_energy_internal": data["potential_energy_internal"],
        "omega": omega,
        "times": times,
        "unit_energy": unit_energy,
        "history_stride": history_stride,
        "bins": bins,
        "norm": norm,
        "cmap": cmap,
        "fig": fig,
        "histogram_axis": histogram_axis,
        "evolution_axis": evolution_axis,
        "bars": bars,
        "line_collection": line_collection,
        "current_points": current_points,
        "time_label": time_label,
        "outdir": outdir,
        "dpi": dpi,
    }
    _W.update(worker)
    atexit.register(close_worker)


def close_worker():
    file = _W.get("file")
    if file is not None:
        file.close()
    fig = _W.get("fig")
    if fig is not None:
        plt.close(fig)


def render(job):
    frame_number, time_index = job
    worker = _W
    history_indices = numpy.arange(0, time_index, worker["history_stride"])
    read_indices = numpy.append(history_indices, time_index)
    energies = read_jacobii(worker, read_indices)
    history_energies = energies[:-1]
    current_energies = energies[-1]
    unit_energy = worker["unit_energy"]

    if history_indices.size:
        times = worker["times"][history_indices]
        segments = numpy.stack(
            (
                numpy.broadcast_to(times[:, None], history_energies.shape),
                unit_energy * history_energies,
            ),
            axis=2,
        ).transpose(1, 0, 2)
    else:
        segments = numpy.empty((history_energies.shape[1], 0, 2))
    worker["line_collection"].set_segments(segments)

    values = unit_energy * current_energies
    counts, _ = numpy.histogram(values, bins=worker["bins"])
    for bar, count in zip(worker["bars"], counts):
        bar.set_height(count)
    worker["current_points"].set_offsets(
        numpy.column_stack((numpy.full(values.shape, worker["times"][time_index]), values))
    )
    worker["time_label"].set_text("{:.1f} Myr".format(1000 * worker["times"][time_index]))
    worker["fig"].tight_layout()
    worker["fig"].savefig(
        worker["outdir"] / "frame_{:06d}.png".format(frame_number),
        dpi=worker["dpi"],
    )
    return frame_number


def main(path, group=None, stride=100, history_stride=100, dpi=300, nproc=None,
         single_frame=False, output_dir=None):
    if stride < 1 or history_stride < 1:
        raise ValueError("stride and history_stride must be positive")
    if nproc is not None and nproc < 1:
        raise ValueError("nproc must be positive")

    path = pathlib.Path(path)
    with h5py.File(path, "r") as file:
        if group is None:
            groups = [name for name in file if name.startswith("nbody")]
            if len(groups) != 1:
                raise SystemExit("choose a group: {}".format(groups))
            group = groups[0]

        data = file[group]
        nframes = data["time"].shape[0]
        mass = data["mass"][:]
        omega = file["schwarzschild"].attrs["omega"]
        jacobii_min = numpy.inf
        jacobii_max = -numpy.inf
        for time_index in range(nframes):
            trajectories = data["trajectories"][time_index]
            energy = (
                mass * data["potential_energy_external"][time_index]
                + 0.5 * mass * data["potential_energy_internal"][time_index]
                + 0.5 * mass * numpy.sum(trajectories[:, 3:6] ** 2, axis=1)
            )
            angular_momentum_z = (
                trajectories[:, 0] * trajectories[:, 4]
                - trajectories[:, 1] * trajectories[:, 3]
            )
            jacobii = energy - omega * angular_momentum_z
            jacobii_min = min(jacobii_min, float(jacobii.min()))
            jacobii_max = max(jacobii_max, float(jacobii.max()))

        unit_energy = data.attrs["unitV"] ** 2 / 1e3

    indices = list(range(0, nframes, stride))
    if not indices:
        raise ValueError("the input contains no frames")
    if single_frame:
        indices = indices[:1]

    outdir = pathlib.Path(output_dir) if output_dir is not None else path.with_suffix("") / group
    outdir.mkdir(parents=True, exist_ok=True)
    bins = unit_energy * numpy.linspace(
        jacobii_min, jacobii_max, max(2, int(2 * numpy.sqrt(mass.shape[0])))
    )
    nproc = nproc or max(1, (os.cpu_count() or 1) - 1)
    nproc = min(nproc, len(indices))
    jobs = list(enumerate(indices))

    initargs = (
        path, group, outdir, bins,
        jacobii_min / 2, jacobii_max / 2,
        unit_energy, history_stride, dpi,
    )
    if single_frame:
        try:
            init_worker(*initargs)
            render(jobs[0])
        finally:
            close_worker()
        print("rendered frame in", outdir)
        return

    with Pool(
        nproc,
        initializer=init_worker,
        initargs=initargs,
    ) as pool:
        for completed, _ in enumerate(pool.imap_unordered(render, jobs, chunksize=1), 1):
            if completed % 100 == 0 or completed == len(jobs):
                print("{:d}/{:d}".format(completed, len(jobs)), flush=True)
    print("frames in", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("group", nargs="?", default=None,
                        help="default: the only nbody_* group in the file")
    parser.add_argument("--stride", type=int, default=100,
                        help="render every Nth simulation timestep")
    parser.add_argument("--history-stride", type=int, default=100,
                        help="plot each particle's energy every Nth timestep")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--single-frame", action="store_true",
                        help="render only the first selected frame without multiprocessing")
    parser.add_argument("--output-dir", default=None,
                        help="directory for rendered frames (default: <file>/<group>)")
    args = parser.parse_args()
    main(
        args.file,
        args.group,
        stride=args.stride,
        history_stride=args.history_stride,
        dpi=args.dpi,
        nproc=args.nproc,
        single_frame=args.single_frame,
        output_dir=args.output_dir,
    )
