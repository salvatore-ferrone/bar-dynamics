"""Step 1 of the pipeline: build an equilibrium Schwarzschild model of the bar.

Writes the `/schwarzschild` group of the HDF5 file. It does not sample the DF; that is done by nbody.py.
"""
import configparser
import datetime
import h5py
import agama, numpy
from astropy import units as u

import common

GROUP = "schwarzschild"

DEFAULT_HYPERPARAMETERS = dict(
    # orbital library
    cross_times=50,
    beta=1 / 2,
    kappa=1 / 10,
    norbits=int(1e4),
    seed=0,
    # density targets
    nnodes=11,
    rgridmax=1.2,
    rgridmin=0.1,
    stripsPerPane=2,
    # optimisation
    sigma_frac=0.1,
    sigma_floor_frac=0.05,
    regul=1.0,
)


def main(path_out, path_bar, path_ext, overwrite=False, **overrides):
    hp = {**DEFAULT_HYPERPARAMETERS, **overrides}
    norbits = hp["norbits"]

    # inputs -> physical-unit parameters, units, pattern speed
    parameters = {**common.load_potential_parameters(path_ext), **common.load_potential_parameters(path_bar)}
    barconfig = configparser.ConfigParser()
    barconfig.read(path_bar)
    units = common.make_units(parameters["bar"]["mass"], parameters["bar"]["scaleRadius"])
    common.set_agama_units(units)
    omega_phys = float(barconfig["orientation"]["pattern_speed_kms_kpc"])
    omega = ((omega_phys * (u.km / u.s) / u.kpc).to(units["unitRot"])).value
    bar, potential_external, rescaled = common.build_potentials(parameters, units)
    potential_total = bar + potential_external

    with h5py.File(path_out, "a") as f:
        if GROUP in f:
            if not overwrite:
                raise FileExistsError("'{}' already exists in {}; pass overwrite=True".format(GROUP, path_out))
            del f[GROUP]
        g = f.create_group(GROUP)
        g.attrs["success"] = False  # flipped at the very end so partial files are recognisable
        g.attrs["schema_version"] = common.SCHEMA_VERSION
        g.attrs["created"] = datetime.datetime.now().isoformat()
        g.attrs["git_commit"] = common.git_commit()
        g.attrs["agama_version"] = agama.__version__
        g.attrs["path_bar"] = str(path_bar)
        g.attrs["path_ext"] = str(path_ext)
        g.attrs["omega"] = omega
        common.write_units(g, units)
        common.write_attrs(g, hp)
        common.write_potential(g, parameters, omega_phys,
                               {"bar_ini": open(path_bar).read(), "ext_ini": open(path_ext).read()})

        # initial conditions of the orbit library
        numpy.random.seed(hp["seed"])
        agama.setRandomSeed(hp["seed"])
        ics, _ = bar.sample(norbits, potential=potential_total, beta=hp["beta"], kappa=hp["kappa"])
        times = hp["cross_times"] * potential_total.Tcirc(ics)

        # density targets
        gridr = agama.nonuniformGrid(hp["nnodes"], hp["rgridmin"], hp["rgridmax"])
        target_density = agama.Target(type="DensityClassicLinear",
                                      gridr=gridr,
                                      stripsPerPane=hp["stripsPerPane"],
                                      axisRatioY=parameters["bar"]["axisRatioY"],
                                      axisRatioZ=parameters["bar"]["axisRatioZ"])

        # orbital library
        D, traj = agama.orbit(ic=ics, potential=potential_total, Omega=omega, dtype=object,
                              time=times, separateTime=True, targets=target_density)

        # solve for the weights
        total_bar_mass = bar.totalMass()
        mass_true = target_density(bar)  # right hand side of the inversion
        sigma = numpy.maximum(hp["sigma_frac"] * mass_true, hp["sigma_floor_frac"] * mass_true.mean())
        weights = agama.solveOpt(
            matrix=[numpy.ones((1, norbits)), D.T],
            rhs=[[total_bar_mass], mass_true],
            rpenq=[[numpy.inf], 2 * sigma ** -2],
            xpenq=hp["regul"] * 2 * numpy.ones(norbits) * norbits / total_bar_mass ** 2,
        )
        mass_model = D.T.dot(weights)

        # save: inputs, library, solution, diagnostics
        g.attrs["total_bar_mass"] = total_bar_mass
        g["ics"] = ics
        g["orbit_times"] = times
        g["gridr"] = gridr
        g["matrix_data_density"] = D
        g["mass_true"] = mass_true
        g["sigma"] = sigma
        g["weights"] = weights
        g["mass_model"] = mass_model

        # ragged orbits stored as concatenated samples + offsets
        # (each orbit is evaluated at its own recorded time points)
        orbit_t = [numpy.array([orb[i] for i in range(len(orb))]) for orb in traj]
        lib = g.create_group("trajectories")
        lib.create_dataset("data", data=numpy.concatenate([orb(t) for orb, t in zip(traj, orbit_t)]),
                           compression="gzip")
        lib.create_dataset("time", data=numpy.concatenate(orbit_t), compression="gzip")
        lib.create_dataset("offsets", data=numpy.cumsum([0] + [len(t) for t in orbit_t]))

        rel_resid = (mass_model - mass_true) / mass_true
        g.attrs["n_nonzero_weights"] = int(numpy.count_nonzero(weights))
        g.attrs["max_abs_rel_residual"] = float(numpy.max(numpy.abs(rel_resid)))
        g.attrs["total_mass_model"] = float(weights.sum())
        g.attrs["success"] = True

    print("saved", GROUP, "->", path_out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="ferrers_bar.h5")
    parser.add_argument("--bar", default="/Users/ferrone/repos/bar-dynamics/parameters/components/bar_ferrers_v1.ini")
    parser.add_argument("--ext", default="/Users/ferrone/repos/bar-dynamics/parameters/components/mw_fiducial_v1.ini")
    parser.add_argument("--overwrite", action="store_true")
    for key, value in DEFAULT_HYPERPARAMETERS.items():
        parser.add_argument("--" + key, type=type(value), default=value)
    args = vars(parser.parse_args())
    main(args.pop("out"), args.pop("bar"), args.pop("ext"), args.pop("overwrite"), **args)
