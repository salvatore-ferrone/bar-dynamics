"""Render PNG frames of an nbody group in parallel.

usage: python animate_single_sim.py ferrers_bar.h5 nbody_200 [--stride 10] [--dpi 150]
then:  ffmpeg -framerate 30 -i ferrers_bar/nbody_200/frame_%06d.png -pix_fmt yuv420p out.mp4
"""
import argparse
import os
import pathlib
from multiprocessing import Pool

import agama
import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy
from astropy import units as u

import common

_W = {}  # per-process state, filled by init_worker


def disk_background(path, extent, npoints):
    """Projected disk density on a grid, in code units."""
    with h5py.File(path, "r") as f:
        units, parameters, _ = common.read_potential(f["schwarzschild"])
    common.set_agama_units(units)
    disks = {k: v for k, v in parameters.items() if "disk" in k}
    rescaled = common.rescale_ini_dict(disks, units["unitM"], units["unitL"])
    pot = agama.Potential(*rescaled.values())
    xs = numpy.linspace(-extent, extent, npoints)
    X, Y = numpy.meshgrid(xs, xs)
    density = pot.projectedDensity(numpy.column_stack((X.ravel(), Y.ravel())))
    return X, Y, density.reshape(X.shape)


def init_worker(path, group, outdir, background, extent, size, dpi, point_size):
    f = h5py.File(path, "r")
    g = f[group]
    t_myr = (g["time"][:] * g.attrs["unitT"] * u.kpc / (u.km / u.s)).to(u.Myr).value

    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0, 0, 1, 1])
    X, Y, density = background
    ax.pcolormesh(X, Y, density, cmap="gray", shading="gouraud")
    ax.set(xlim=[-extent, extent], ylim=[-extent, extent], aspect="equal", xticks=[], yticks=[])
    ax.text(0.01, 0.99, "N = {:d}".format(g["mass"].shape[0]), va="top", ha="left",
            transform=ax.transAxes, color="white")
    txt = ax.text(0.99, 0.99, "", va="top", ha="right", transform=ax.transAxes, color="white")
    scat = ax.scatter([], [], s=point_size, color="k")

    _W.update(traj=g["trajectories"], t=t_myr, fig=fig, scat=scat, txt=txt, outdir=outdir, dpi=dpi)


def render(i):
    w = _W
    w["scat"].set_offsets(w["traj"][i, :, 0:2])
    w["txt"].set_text("{:.0f} Myr".format(w["t"][i]))
    w["fig"].savefig(w["outdir"] / "frame_{:06d}.png".format(i), dpi=w["dpi"])
    return i


def main(path, group, stride=1, dpi=150, extent=2.0, size=6.0, npoints=200, point_size=2, nproc=None):
    path = pathlib.Path(path)
    outdir = path.with_suffix("") / group
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as f:
        nframes = f[group]["trajectories"].shape[0]
    indices = range(0, nframes, stride)
    nproc = nproc or max(1, os.cpu_count() - 1)
    background = disk_background(path, extent, npoints)

    with Pool(nproc, initializer=init_worker,
              initargs=(path, group, outdir, background, extent, size, dpi, point_size)) as pool:
        for n, _ in enumerate(pool.imap_unordered(render, indices, chunksize=8), 1):
            if n % 100 == 0 or n == len(indices):
                print("{:d}/{:d}".format(n, len(indices)), flush=True)
    print("frames in", outdir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("file")
    p.add_argument("group", nargs="?", default=None, help="default: the only nbody_* group")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--extent", type=float, default=2.0, help="half-width of the view, code units")
    p.add_argument("--nproc", type=int, default=None)
    a = p.parse_args()
    group = a.group
    if group is None:
        with h5py.File(a.file, "r") as f:
            groups = [k for k in f if k.startswith("nbody")]
        if len(groups) != 1:
            raise SystemExit("choose a group: {}".format(groups))
        group = groups[0]
    main(a.file, group, a.stride, a.dpi, a.extent, nproc=a.nproc)