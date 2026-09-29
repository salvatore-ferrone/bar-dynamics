#!/usr/bin/env python3
"""Generate a Palomar 5 particle-spray stream in a rotating Milky Way bar."""

import argparse
from pathlib import Path

import agama
import matplotlib.pyplot as plt
import numpy as np


# Pal 5 present-day phase-space estimate: degrees, kpc, mas/yr, km/s.
PAL5_ICRS = (229.019167, -0.121, 22.0041224, -2.76492231, -2.64181048, -58.6298157)
PAL5_MASS_MSUN = 2.5e4
BAR_ANGLE_RAD = np.deg2rad(-25.0)
TIME_UNIT_GYR = 0.977792221


def make_potential(pattern_speed):
	"""Build a disk+halo Milky Way model with a rotating triaxial Ferrers bar."""
	disk_thin = agama.Potential(
		type="MiyamotoNagai", mass=6.0e10, scaleRadius=3.5, scaleHeight=0.28
	)
	disk_thick = agama.Potential(
		type="MiyamotoNagai", mass=1.0e10, scaleRadius=2.0, scaleHeight=0.8
	)
	halo = agama.Potential(type="NFW", mass=1.0e12, scaleRadius=20.0)
	axisymmetric = agama.Potential(disk_thin, disk_thick, halo)
	bar = agama.Potential(
		type="Ferrers", mass=1.0e10, scaleRadius=3.5, axisRatioY=0.35, axisRatioZ=0.25
	)
	fixed_model = agama.Potential(axisymmetric, bar)
	rotating_model = agama.Potential(
		potential=fixed_model,
		rotation=[[0.0, BAR_ANGLE_RAD], [1.0, BAR_ANGLE_RAD + pattern_speed]],
	)
	return axisymmetric, rotating_model


def pal5_galactocentric_state():
	ra, dec, distance, pmra, pmdec, radial_velocity = PAL5_ICRS
	lon, lat, pm_lon, pm_lat = agama.transformCelestialCoords(
		agama.fromICRStoGalactic, np.deg2rad(ra), np.deg2rad(dec), pmra, pmdec
	)
	return np.asarray(
		agama.getGalactocentricFromGalactic(
			lon, lat, distance, pm_lon * 4.74047, pm_lat * 4.74047, radial_velocity
		),
		dtype=float,
	)


def particle_spray(potential, axisymmetric, progenitor_now, mass, lookback_gyr, count, seed):
	if count < 4 or count % 2:
		raise ValueError("--particles must be an even integer of at least 4")

	lookback = lookback_gyr / TIME_UNIT_GYR
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
	derivatives = axisymmetric.eval(orbit[:, :3], der=True)
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


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--bar-pattern-speed", type=float, default=-39.0, help="km/s/kpc")
	parser.add_argument("--lookback-gyr", type=float, default=4.0)
	parser.add_argument("--particles", type=int, default=512, help="even count, split between arms")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
	args = parser.parse_args()

	agama.setUnits(length=1, velocity=1, mass=1)
	axisymmetric, potential = make_potential(args.bar_pattern_speed)
	progenitor_now = pal5_galactocentric_state()
	orbit, stream = particle_spray(
		potential, axisymmetric, progenitor_now, PAL5_MASS_MSUN,
		args.lookback_gyr, args.particles, args.seed
	)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	tag = f"Omega_{args.bar_pattern_speed:g}"
	data_path = args.output_dir / f"pal5_stream_{tag}.npz"
	plot_path = args.output_dir / f"pal5_stream_{tag}.png"
	np.savez_compressed(
		data_path,
		stream_xv=stream,
		progenitor_xv=progenitor_now,
		progenitor_orbit=orbit,
		bar_pattern_speed_kms_kpc=args.bar_pattern_speed,
		lookback_gyr=args.lookback_gyr,
		pal5_icrs=np.asarray(PAL5_ICRS),
	)

	fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=140)
	axes[0].scatter(stream[:, 0], stream[:, 1], s=3, alpha=0.65)
	axes[0].scatter(progenitor_now[0], progenitor_now[1], marker="*", s=70, c="black")
	axes[0].set(xlabel="Galactocentric x [kpc]", ylabel="y [kpc]", title="Palomar 5 stream")
	axes[0].set_aspect("equal", adjustable="datalim")
	axes[1].scatter(stream[:, 0], stream[:, 2], s=3, alpha=0.65)
	axes[1].scatter(progenitor_now[0], progenitor_now[2], marker="*", s=70, c="black")
	axes[1].set(xlabel="Galactocentric x [kpc]", ylabel="z [kpc]", title="Vertical structure")
	fig.suptitle(f"Ferrers bar pattern speed: {args.bar_pattern_speed:g} km/s/kpc")
	fig.tight_layout()
	fig.savefig(plot_path)
	print(f"Generated {len(stream)} Pal 5 stream particles.")
	print(f"Data: {data_path}")
	print(f"Plot: {plot_path}")


if __name__ == "__main__":
	main()
