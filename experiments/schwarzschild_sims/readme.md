# Schwarzschild → N-body pipeline

Run everything from `experiments/schwarzschild_sims/` in the `bar-dynamics` conda env:

```bash
conda activate bar-dynamics
cd experiments/schwarzschild_sims
```

Both stages read and write one HDF5 file, with a group for each stage.

## 1. Build the equilibrium model

```bash
python schwarzschild.py --out ferrers_bar.h5
```

Useful options (defaults are in `DEFAULT_HYPERPARAMETERS`):

```bash
python schwarzschild.py --out ferrers_bar.h5 \
    --bar ../../parameters/components/bar_ferrers_v1.ini \
    --ext ../../parameters/components/mw_fiducial_v1.ini \
    --norbits 10000 --cross_times 50 --seed 1 --overwrite
```

This writes `/schwarzschild`. Re-running into an existing group fails unless you pass `--overwrite`.

## 2. Sample and integrate an N-body run

```bash
python nbody.py ferrers_bar.h5 --Nbody 100000
```

Options:

```bash
python nbody.py ferrers_bar.h5 --Nbody 100000 --seed 2 \
    --treeAngle 0.1 --softening_frac 0.1 \
    --integrationtime_Gyr 1.0 --samplingtime_Myr 0.1 \
    --timestep_frac 0.02 --group nbody_100000_theta0p1 --overwrite
```

This writes `/nbody_<N>` (or `--group`). Use a different `--group` for each hyperparameter variation so runs don't collide.

## 3. From Python

```python
import schwarzschild, nbody

schwarzschild.main("ferrers_bar.h5", 
                   "../../parameters/components/bar_ferrers_v1.ini",
                   "../../parameters/components/mw_fiducial_v1.ini",
                   norbits=2000, overwrite=True)

nbody.main("ferrers_bar.h5", Nbody=10000, integrationtime_Gyr=0.5)
```

## 4. Reading the results

```python
import h5py

with h5py.File("ferrers_bar.h5", "r") as f:
    f.visit(print)                       # list everything

    s = f["schwarzschild"]
    print(s.attrs["success"], s.attrs["max_abs_rel_residual"])
    weights = s["weights"][:]
    mass_true, mass_model = s["mass_true"][:], s["mass_model"][:]

    g = f["nbody_10000"]
    traj = g["trajectories"]             # (nsampling+1, Nbody, 6), float32, lazily read
    t = g["time"][:]                     # code units
    snapshot = traj[-1]                  # read a single snapshot
    print(dict(g.attrs))                 # all hyperparameters actually used
```

## Units

Every group stores `unitM`, `unitL`, `unitV`, `unitT` and `unitG` in (Msun, kpc, km/s, kpc/(km/s)).
To convert a stored number to physical units, multiply by the unit:

```python
import astropy.units as u
with h5py.File("ferrers_bar.h5", "r") as f:
    g = f["nbody_10000"]
    t_Myr = (g["time"][:] * g.attrs["unitT"] * u.kpc / (u.km / u.s)).to(u.Myr)
```

## Reproducing a run from the file alone

`/schwarzschild/potential` and `/schwarzschild/inputs` hold the physical potential parameters and the raw ini text.
`nbody.py` rebuilds everything from these, so the original ini paths are not needed.
Seeds are stored in the attributes.