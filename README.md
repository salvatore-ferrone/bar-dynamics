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
- `simulations/`: one HDF5 file per run, including phase-space arrays, metadata, and snapshots of both input files.
- `analysis/` and `plots/`: analysis scripts and quick-look figures.

Run the 71 clockwise bar speeds from 25 to 60 km/s/kpc, plus the matched axisymmetric control, with:

```bash
python experiments/simple_example.py
```

The INI stores positive pattern-speed magnitudes; clockwise Agama pattern speeds are signed negative. The control retains the azimuthally averaged (`m=0`) contribution of the same Ferrers bar while keeping the disk and halo unchanged and static.

Create the morphology overlay with:

```bash
python analysis/quicklook.py
```

Output names encode the stream, Milky Way model ID, lookback time, particle count, random seed, and input snapshot hash; barred runs also encode the bar ID and pattern speed. Each HDF5 file contains `stream/phase_space`, progenitor and spray data, units, software metadata, and the complete text inputs used for that run.