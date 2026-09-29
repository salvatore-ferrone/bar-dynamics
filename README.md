# Bar Dynamics: a pedagogical primer

How does the galactic bar effect stellar streams
1. Consider globular cluster streams
2. Full 3D motion
  - Not restricted to planar orbits
  - Not restricted to nearly circular orbits 
3. Can we infer bar properties from streams? 
4. Which streams?

## First Experiment: Palomar 5 Bar-Speed Sweep

The initial experiment uses four folders:

- `parameters/`: Pal 5 phase-space data and the INI configuration for the Milky Way model and sweep.
- `experiments/`: Agama simulation drivers.
- `simulations/`: one merged HDF5 file per experiment, with one group per run.
- `analysis/` and `plots/`: analysis scripts and quick-look figures.

Run the 71 clockwise bar speeds from 25 to 60 km/s/kpc, plus the matched axisymmetric control, with:

```bash
python experiments/bar_pattern_speed_sweep.py
```

The INI stores positive pattern-speed magnitudes; clockwise Agama pattern speeds are signed negative. The control retains the azimuthally averaged (`m=0`) contribution of the same Ferrers bar while keeping the disk and halo unchanged and static.

Each run is first written to its own temporary HDF5 file (independent, so future runs can be parallelized safely), then all runs are merged into a single packaged HDF5 file under `simulations/`, and the temp files are deleted. The merged file stores shared metadata (config, initial conditions, units, software versions) at the top level, plus one group per run under `runs/<run_key>` containing that run's progenitor orbit (`progenitor_orbit_phase_space`) and final stream (`stream_phase_space`).

Create the quick-look overlay and the per-speed figures with:

```bash
python analysis/quicklook.py
python analysis/pattern_speed_sweep.py
```