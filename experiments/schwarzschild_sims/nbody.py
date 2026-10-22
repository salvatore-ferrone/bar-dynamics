"""Step 2 of the pipeline: sample an N-body realisation of the Schwarzschild model and integrate it.

Reads `/schwarzschild` from the HDF5 file (potential, units and orbit weights all come from the file,
never from ini files) and writes a new `/nbody_<N>` group to the same file.
"""
import datetime
import h5py
import agama, numpy
import pyfalcon
from astropy import units as u

import common
from schwarzschild import GROUP as SCHWARZSCHILD_GROUP

DEFAULT_HYPERPARAMETERS = dict(
    Nbody=int(1e2),
    seed=0,
    virial_tolerence=1 / 10,
    treeAngle=0.1,            # 0 means exact n body summation
    softening_frac=1e-1,      # softening as a fraction of the mean inter-particle distance
    integrationtime_Gyr=1.0,  # time step and sampling time are adjusted to be evenly spaced
    timestep_frac=1 / 50,
    samplingtime_Myr=0.1,
)
NPHASE = 6


def main(path_file, group=None, overwrite=False, **overrides):
    hp = {**DEFAULT_HYPERPARAMETERS, **overrides}
    nbody = hp["Nbody"]
    group = group or "nbody_{:d}".format(nbody)

    with h5py.File(path_file, "a") as f:
        src = f[SCHWARZSCHILD_GROUP]
        if not src.attrs["success"]:
            raise RuntimeError("the schwarzschild model in {} did not finish successfully".format(path_file))
        if group in f:
            if not overwrite:
                raise FileExistsError("'{}' already exists in {}; pass overwrite=True or choose another group".format(group, path_file))
            del f[group]

        # rebuild the physical setup from the file alone
        units, parameters, _ = common.read_potential(src)
        unitT = units["unitT"]
        common.set_agama_units(units)
        bar, potential_external, rescaled = common.build_potentials(parameters, units)

        # sample the initial conditions from the orbit library
        numpy.random.seed(hp["seed"])
        # agama needs Orbit objects, so the (deterministic) library is re-integrated from the stored ICs
        orbits = agama.orbit(ic=src["ics"][:], potential=bar + potential_external, Omega=src.attrs["omega"],
                             dtype=object, time=src["orbit_times"][:], separateTime=True)
        snap_posvel, snap_mass = agama.sampleOrbitLibrary(nbody, orbits, src["weights"][:])
        virial_ratio = compute_virial_ratio(snap_posvel, snap_mass, potential_external)
        if numpy.abs(1 - virial_ratio) > hp["virial_tolerence"]:
            print("Warning, virial ratio is off by more than {:.0%}. It is {:.3e}".format(hp["virial_tolerence"], virial_ratio))

        # softening from the mean inter-particle distance
        p = rescaled["bar"]
        barvolume = numpy.pi * p["scaleRadius"] ** 3 * p["axisRatioY"] * p["axisRatioZ"]
        eps = hp["softening_frac"] * (barvolume / nbody) ** (1 / 3)

        # adjust the time stepping
        T = (hp["integrationtime_Gyr"] * u.Gyr).to(unitT).value
        ts = (hp["samplingtime_Myr"] * u.Myr).to(unitT).value
        dt_max = hp["timestep_frac"] * numpy.sqrt(eps ** 3 / snap_mass.mean())
        nsampling = int(round(T / ts))
        steps_per_sample = int(numpy.ceil((T / nsampling) / dt_max))
        nsteps = nsampling * steps_per_sample
        dt = T / nsteps
        samplingtime = steps_per_sample * dt
        data_volume = (nbody * NPHASE * (nsampling + 1) * 32 * u.bit).to(u.Gbyte)
        print("{:<25s} {:.2e}".format("integration time", (nsteps * dt * unitT).to(u.Myr)))
        print("{:<25s} {:.2e}".format("dt", (dt * unitT).to(u.Myr)))
        print("{:<25s} {:.2e}".format("sampling", (samplingtime * unitT).to(u.Myr)))
        print("{:<25s} {:d}".format("nsampling", nsampling))
        print("{:<25s} {:d}".format("nsteps", nsteps))
        print("{:<25s} {:.2e}".format("Data volume", data_volume))

        # output group
        g = f.create_group(group)
        g.attrs["success"] = False  # flipped at the very end
        g.attrs["schema_version"] = common.SCHEMA_VERSION
        g.attrs["created"] = datetime.datetime.now().isoformat()
        g.attrs["git_commit"] = common.git_commit()
        g.attrs["source_group"] = SCHWARZSCHILD_GROUP
        common.write_units(g, units)
        common.write_attrs(g, hp)
        g.attrs.update(eps=eps, dt=dt, nsteps=nsteps, nsampling=nsampling, steps_per_sample=steps_per_sample,
                       samplingtime=samplingtime, integrationtime=nsteps * dt, virial_ratio=virial_ratio,
                       data_volume_Gbyte=data_volume.value)
        g["initial_conditions"] = snap_posvel
        g["mass"] = snap_mass
        g["time"] = numpy.arange(nsampling + 1) * samplingtime
        traj = g.create_dataset("trajectories", (nsampling + 1, nbody, NPHASE), dtype="float32",
                                chunks=(1, nbody, NPHASE), compression="gzip")
        pot_int = g.create_dataset("potential_energy_internal", (nsampling + 1, nbody), dtype="float32",
                                   chunks=(1, nbody), compression="gzip")
        pot_ext = g.create_dataset("potential_energy_external", (nsampling + 1, nbody), dtype="float32",
                                   chunks=(1, nbody), compression="gzip")

        def record(index, pos, vel):
            _, phi_int = pyfalcon.gravity(pos, snap_mass, eps=eps, theta=hp["treeAngle"])
            phi_ext = potential_external.potential(pos)
            traj[index, :, 0:3] = pos
            traj[index, :, 3:6] = vel
            pot_int[index] = phi_int
            pot_ext[index] = phi_ext

        # integrate (kick-drift-kick style, as before)
        pos = snap_posvel[:, 0:3].copy()
        vel = snap_posvel[:, 3:6].copy()
        record(0, pos, vel)
        for i in range(1, nsteps + 1):
            pos += 0.5 * dt * vel
            f_int, _ = pyfalcon.gravity(pos, snap_mass, eps=eps, theta=hp["treeAngle"])
            f_ext = potential_external.force(pos)
            vel += dt * (f_int + f_ext)
            pos += 0.5 * dt * vel
            if i % steps_per_sample == 0:
                record(i // steps_per_sample, pos, vel)
        g.attrs["success"] = True

    print("saved", group, "->", path_file)


def compute_virial_ratio(snap_posvel, snap_mass, potential_external):
    # approximate the self-potential with a multipole expansion
    pot_stars = agama.Potential(type="multipole", particles=(snap_posvel, snap_mass), symmetry="t")
    E_kin = 0.5 * numpy.sum(snap_mass[:, None] * snap_posvel[:, 3:6] ** 2)
    W_self = 0.5 * numpy.sum(snap_mass * pot_stars.potential(snap_posvel[:, 0:3]))
    force_rest = potential_external.force(snap_posvel[:, 0:3])
    W_ext = numpy.sum(snap_mass * numpy.sum(snap_posvel[:, 0:3] * force_rest, axis=1))
    return -2 * E_kin / (W_self + W_ext)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="HDF5 file containing a successful /schwarzschild group")
    parser.add_argument("--group", default=None, help="output group name (default nbody_<N>)")
    parser.add_argument("--overwrite", action="store_true")
    for key, value in DEFAULT_HYPERPARAMETERS.items():
        parser.add_argument("--" + key, type=type(value), default=value)
    args = vars(parser.parse_args())
    main(args.pop("file"), args.pop("group"), args.pop("overwrite"), **args)
