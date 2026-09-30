#!/usr/bin/env python3
"""Generate a globular-cluster particle-spray stream across a sweep of bar pattern speeds.

Each run (the axisymmetric control plus one per pattern speed) is written to its
own temporary HDF5 file, so runs stay independent (safe for future parallel
execution). Once all runs finish, the temp files are merged into a single
packaged HDF5 file for the whole experiment and the temp files are deleted.

Usage: bar_pattern_speed_sweep.py [path/to/experiment.toml]
"""

import configparser
import hashlib
import sys
import tempfile
import tomllib
from pathlib import Path

import agama
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "parameters" / "experiments" / "pal5_bar_pattern_speed.toml"


def apply_overrides(config, overrides):
	"""Overwrite config[section][key] with any matching entries from the experiment's [overrides] table."""
	for section, keys in overrides.items():
		if config.has_section(section):
			for key, value in keys.items():
				config.set(section, key, str(value))


def load_experiment(toml_path):
	with open(toml_path, "rb") as handle:
		experiment = tomllib.load(handle)
	base = toml_path.parent
	components = experiment["components"]
	overrides = experiment.get("overrides", {})

	mw_ini_path = (base / components["milky_way"]).resolve()
	bar_ini_path = (base / components["bar"]).resolve()
	cluster_ini_path = (base / components["cluster"]).resolve()
	initial_conditions_path = (base / components["initial_conditions"]).resolve()

	bar_config = configparser.ConfigParser()
	bar_config.read(bar_ini_path)
	apply_overrides(bar_config, overrides)

	cluster_config = configparser.ConfigParser()
	cluster_config.read(cluster_ini_path)
	apply_overrides(cluster_config, overrides)

	return experiment, mw_ini_path, bar_ini_path, cluster_ini_path, bar_config, cluster_config, initial_conditions_path


def make_potential(mw_ini_path, bar_ini_path, bar_config):
	axisymmetric = agama.Potential(str(mw_ini_path))
	bar = agama.Potential(str(bar_ini_path))
	bar_axisymmetric = agama.Potential(
		type="CylSpline", potential=bar, mmax=0, gridsizeR=30, gridsizez=32,
		Rmin=0.1, Rmax=40, zmin=0.05, zmax=20,
	)
	tidal_potential = agama.Potential(axisymmetric, bar_axisymmetric)
	bar_angle = np.deg2rad(bar_config.getfloat("orientation", "present_angle_deg"))
	return tidal_potential, axisymmetric, bar, bar_angle


def progenitor_galactocentric_state(initial_conditions_path):
	ra, dec, distance, pmra, pmdec, radial_velocity = np.loadtxt(
		initial_conditions_path, delimiter=",", comments="#", ndmin=2
	)[0]
	lon, lat, pm_lon, pm_lat = agama.transformCelestialCoords(
		agama.fromICRStoGalactic, np.deg2rad(ra), np.deg2rad(dec), pmra, pmdec
	)
	return np.asarray(
		agama.getGalactocentricFromGalactic(
			lon, lat, distance, pm_lon * 4.74047, pm_lat * 4.74047, radial_velocity
		),
		dtype=float,
	)


def particle_spray(potential, tidal_potential, progenitor_now, mass, lookback_gyr, count, seed):
	if count < 4 or count % 2:
		raise ValueError("particle count must be an even integer of at least 4")

	lookback = lookback_gyr / (agama.getUnits()["time"] / 1e3)
	release_count = count // 2
	release_times, orbit = agama.orbit(
		potential=potential,
		ic=progenitor_now,
		time=-lookback,
		trajsize=release_count + 1,
		dtype="float64",
	)
	release_times = release_times[1:][::-1]
	orbit = orbit[1:][::-1]

	x, y, z, vx, vy, vz = orbit.T
	radius_sq = x * x + y * y + z * z
	angular_momentum = np.cross(orbit[:, :3], orbit[:, 3:])
	angular_frequency = np.linalg.norm(angular_momentum, axis=1) / radius_sq
	derivatives = tidal_potential.eval(orbit[:, :3], der=True)
	radial_second_derivative = -(
		x*x * derivatives[:, 0] + y*y * derivatives[:, 1] + z*z * derivatives[:, 2]
		+ 2*x*y * derivatives[:, 3] + 2*y*z * derivatives[:, 4] + 2*z*x * derivatives[:, 5]
	) / radius_sq
	tidal_term = angular_frequency**2 - radial_second_derivative
	if np.any(tidal_term <= 0):
		raise ValueError("Could not calculate a real Jacobi radius along this orbit")
	jacobi_radius = (agama.G * mass / tidal_term) ** (1.0 / 3.0)
	release_speed = angular_frequency * jacobi_radius

	# Local radial/tangential/normal frames and warm offsets at both Lagrange points.
	radial = orbit[:, :3] / np.linalg.norm(orbit[:, :3], axis=1)[:, None]
	normal = angular_momentum / np.linalg.norm(angular_momentum, axis=1)[:, None]
	tangential = np.cross(normal, radial)
	frame = np.repeat(np.stack((radial, tangential, normal), axis=1), 2, axis=0)
	signed_radius = np.repeat(jacobi_radius, 2) * np.tile([1.0, -1.0], release_count)
	signed_speed = np.repeat(release_speed, 2) * np.tile([1.0, -1.0], release_count)

	rng = np.random.default_rng(seed)
	radial_offset = (rng.normal(size=count) * 0.5 + 2.0) * signed_radius
	vertical_offset = rng.normal(size=count) * 0.5 * signed_radius
	tangential_speed = (rng.normal(size=count) * 0.5 + 0.3) * signed_speed * (radial_offset / signed_radius)
	vertical_speed = rng.normal(size=count) * 0.5 * signed_speed
	position_offset = np.column_stack((radial_offset, np.zeros(count), vertical_offset))
	velocity_offset = np.column_stack((np.zeros(count), tangential_speed, vertical_speed))
	initial = np.repeat(orbit, 2, axis=0)
	initial[:, :3] += np.einsum("ni,nij->nj", position_offset, frame)
	initial[:, 3:] += np.einsum("ni,nij->nj", velocity_offset, frame)

	seed_times = np.repeat(release_times, 2)
	stream = agama.orbit(
		potential=potential,
		ic=initial,
		timestart=seed_times,
		time=-seed_times,
		trajsize=1,
		separateTime=True,
		dtype="float64",
	)[1].reshape(count, 6)
	return orbit, stream


def save_run_to_temp(temp_dir, run_key, run_type, pattern_speed_magnitude, pattern_speed_signed, orbit, stream):
	"""Write one run's progenitor orbit and final stream to its own temp HDF5 file."""
	path = temp_dir / f"{run_key}.h5"
	with h5py.File(path, "w") as output:
		output.attrs["run_key"] = run_key
		output.attrs["run_type"] = run_type
		output.attrs["bar_pattern_speed_magnitude_kms_kpc"] = pattern_speed_magnitude
		output.attrs["bar_pattern_speed_signed_kms_kpc"] = pattern_speed_signed
		output.create_dataset("progenitor_orbit_phase_space", data=orbit, compression="gzip")
		output.create_dataset("stream_phase_space", data=stream, compression="gzip")
	return path


def merge_runs(temp_paths, output_path, experiment, mw_model_id, bar_model_id, cluster_config, bar_config,
		direction, texts, progenitor_now, lookback_gyr, count, seed):
	"""Combine the independent per-run temp files into one packaged experiment file."""
	with h5py.File(output_path, "w") as output:
		output.attrs["schema_version"] = "2.0"
		output.attrs["experiment_id"] = experiment["experiment"]["name"]
		output.attrs["stream_id"] = cluster_config.get("cluster", "stream_id")
		output.attrs["milky_way_model_id"] = mw_model_id
		output.attrs["bar_model_id"] = bar_model_id
		output.attrs["rotation_direction"] = direction
		output.attrs["bar_present_angle_deg"] = bar_config.getfloat("orientation", "present_angle_deg")
		output.attrs["cluster_mass_msun"] = cluster_config.getfloat("cluster", "mass_msun")
		output.attrs["lookback_gyr"] = lookback_gyr
		output.attrs["particle_count"] = count
		output.attrs["random_seed"] = seed
		output.attrs["agama_version"] = getattr(agama, "__version__", "unknown")
		output.attrs["numpy_version"] = np.__version__
		output.attrs["h5py_version"] = h5py.__version__
		output.attrs["units"] = "position: kpc; velocity: km/s; mass: Msun; time: Gyr"
		output.attrs["phase_space_column_order"] = "x,y,z,vx,vy,vz"
		output.attrs["stream_generation_method"] = "Fardal et al. 2015 particle spray"
		output.attrs["orbit_integrator"] = "Agama default (dop853), relative accuracy 1e-8"
		output.attrs["axisymmetric_control"] = (
			"Static m=0 CylSpline azimuthal average of the same Ferrers bar; "
			"disk and halo identical and static"
		)
		output.attrs["experiment_toml"] = texts["experiment"]
		output.attrs["milky_way_potential_ini"] = texts["milky_way"]
		output.attrs["bar_potential_ini"] = texts["bar"]
		output.attrs["cluster_ini"] = texts["cluster"]
		output.attrs["initial_conditions_txt"] = texts["initial_conditions"]
		output.create_dataset("progenitor_present_phase_space", data=progenitor_now)

		runs = output.create_group("runs")
		for temp_path in temp_paths:
			with h5py.File(temp_path, "r") as temp_file:
				run_key = temp_file.attrs["run_key"]
				run_group = runs.create_group(run_key)
				for attr_key, attr_value in temp_file.attrs.items():
					if attr_key != "run_key":
						run_group.attrs[attr_key] = attr_value
				temp_file.copy("progenitor_orbit_phase_space", run_group)
				temp_file.copy("stream_phase_space", run_group)


def main():
	toml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXPERIMENT

	experiment, mw_ini_path, bar_ini_path, cluster_ini_path, bar_config, cluster_config, initial_conditions_path = (
		load_experiment(toml_path)
	)
	texts = {
		"experiment": toml_path.read_text(),
		"milky_way": mw_ini_path.read_text(),
		"bar": bar_ini_path.read_text(),
		"cluster": cluster_ini_path.read_text(),
		"initial_conditions": initial_conditions_path.read_text(),
	}
	input_hash = hashlib.sha256("\n".join(texts.values()).encode("utf-8")).hexdigest()[:8]

	sweep = experiment["experiment"]["pattern_speed_sweep"]
	minimum = sweep["min_kms_kpc"]
	maximum = sweep["max_kms_kpc"]
	step = sweep["step_kms_kpc"]
	if step <= 0 or maximum < minimum:
		raise SystemExit("Pattern-speed range requires step > 0 and maximum >= minimum")
	steps = (maximum - minimum) / step
	if not np.isclose(steps, round(steps)):
		raise SystemExit("Pattern-speed range must be evenly divisible by its step")
	speeds = np.round(minimum + step * np.arange(round(steps) + 1), decimals=10)

	lookback_gyr = experiment["experiment"]["lookback_gyr"]
	count = experiment["experiment"]["particle_count"]
	seed = experiment["experiment"]["random_seed"]
	cluster_mass = cluster_config.getfloat("cluster", "mass_msun")
	direction = sweep["rotation_direction"].lower()
	if direction not in {"clockwise", "counterclockwise"}:
		raise SystemExit("rotation_direction must be clockwise or counterclockwise")
	direction_sign = -1.0 if direction == "clockwise" else 1.0
	direction_tag = "cw" if direction == "clockwise" else "ccw"

	agama.setUnits(length=1, velocity=1, mass=1)
	tidal_potential, background, bar, bar_angle = make_potential(mw_ini_path, bar_ini_path, bar_config)
	progenitor_now = progenitor_galactocentric_state(initial_conditions_path)

	name = experiment["experiment"]["name"]
	mw_model_id = mw_ini_path.stem
	bar_model_id = bar_ini_path.stem
	lookback_tag = f"{lookback_gyr:g}".replace(".", "p")
	experiment_name = (
		f"{name}__mw-{mw_model_id}__lookback-{lookback_tag}gyr__n-{count}__"
		f"seed-{seed:04d}__{input_hash}"
	)

	runs = [("axisymmetric_control", "axisymmetric_control", 0.0, 0.0, tidal_potential)]
	for speed in speeds:
		rotating_bar = agama.Potential(
			potential=bar,
			rotation=[[0.0, bar_angle], [1.0, bar_angle + direction_sign * speed]],
		)
		model = agama.Potential(background, rotating_bar)
		run_key = f"omega_{direction_tag}_{speed:05.1f}"
		runs.append((run_key, "barred", float(speed), direction_sign * speed, model))

	output_dir = ROOT / "simulations"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{experiment_name}.h5"

	print(f"Running {len(runs)} simulations: one control and {len(speeds)} pattern speeds")
	with tempfile.TemporaryDirectory(prefix=f"{experiment_name}__") as temp_dir_name:
		temp_dir = Path(temp_dir_name)
		temp_paths = []
		for index, (run_key, run_type, magnitude, signed_speed, model) in enumerate(runs, start=1):
			orbit, stream = particle_spray(
				model, tidal_potential, progenitor_now, cluster_mass, lookback_gyr, count, seed
			)
			temp_paths.append(
				save_run_to_temp(temp_dir, run_key, run_type, magnitude, signed_speed, orbit, stream)
			)
			print(f"[{index}/{len(runs)}] {run_key} -> temp file written")

		merge_runs(
			temp_paths, output_path, experiment, mw_model_id, bar_model_id, cluster_config, bar_config,
			direction, texts, progenitor_now, lookback_gyr, count, seed,
		)
	print(f"Merged {len(runs)} runs into {output_path}")


if __name__ == "__main__":
	main()
