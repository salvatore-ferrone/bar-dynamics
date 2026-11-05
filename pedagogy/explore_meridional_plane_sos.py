"""Interactive meridional-plane surface-of-section explorer.

This applet mirrors the interaction style of explore_hunter_bar_frame_sos.py,
but for an axisymmetric potential in the meridional plane (R, z) with fixed Lz
and energy E. Right-click in the SoS panel (R, p_R) to seed an orbit.
"""

from dataclasses import dataclass
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import scipy.optimize
import agama


matplotlib.rc("axes", linewidth=0.6)
matplotlib.rc("font", size=9)

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

agama.setUnits(length=1, mass=1, velocity=1)


# ----------------------------- user parameters -----------------------------
POTENTIAL_FILENAME = "McMillan17.ini"

# Pick Lz from a fraction of Lzmax at the reference energy shell.
REFERENCE_RADIUS_FOR_LZMAX = 10.0
LZ_FRACTION_OF_LZMAX = 0.99
LZ_OVERRIDE = None

# Energy model:
# E_circ(Lz) is the lowest energy at fixed Lz; E_escape is taken as 0 by default
# for this potential normalization. Choose E in-between using ENERGY_FRACTION.
ENERGY_FRACTION = 0.25
E_ESCAPE = 0.0
E_OVERRIDE = None

ORBIT_TIME_MULTIPLIER = 120.0
TRAJSIZE = 120000
ORBIT_TAIL_FRACTION = 0.15
N_GRID = 250

RNG_SEED = 17
ZVC_BRACKET_MAX = 1e4

# Saving of curated initial conditions from the interactive session.
SAVE_IC_FILENAME = "meridional_sos_initial_conditions_{:.2f}_{:.2f}_{:.2f}.txt".format(REFERENCE_RADIUS_FOR_LZMAX, LZ_FRACTION_OF_LZMAX, ENERGY_FRACTION)
SAVE_IC_HEADER = "# x  y  z  vx  vy  vz"
SAVE_FIG_FILENAME = os.path.splitext(SAVE_IC_FILENAME)[0] + ".png"
# --------------------------------------------------------------------------


def mcmillan_potential(filename=POTENTIAL_FILENAME):
	return agama.Potential(agama.__path__[0] + "/data/" + filename)


def lz_circular_at_R(pot, R):
	ax = float(pot.force(np.array([R, 0.0, 0.0]))[0])
	return np.sqrt(max(-R**3 * ax, 0.0))


def rcirc_for_lz(pot, Lz, rmin=1e-3, rmax=200.0):
	def residual(R):
		return lz_circular_at_R(pot, R) - Lz

	return float(scipy.optimize.brentq(residual, rmin, rmax))


def phi_eff_meridional(pot, R, z, Lz):
	R = np.asarray(R, dtype=float)
	z = np.asarray(z, dtype=float)
	pts = np.column_stack((R.ravel(), np.zeros(R.size), z.ravel()))
	return (pot.potential(pts) + 0.5 * (Lz / R.ravel()) ** 2).reshape(R.shape)


def choose_energy_and_lz(pot):
	eref = float(pot.potential(REFERENCE_RADIUS_FOR_LZMAX, 0.0, 0.0))
	lzmax = float(2 * np.pi * pot.Rcirc(E=eref) ** 2 / pot.Tcirc(eref))

	Lz = float(LZ_OVERRIDE) if LZ_OVERRIDE is not None else float(LZ_FRACTION_OF_LZMAX * lzmax)
	rcirc = rcirc_for_lz(pot, Lz)
	ecirc = float(phi_eff_meridional(pot, np.array([rcirc]), np.array([0.0]), Lz)[0])

	if E_OVERRIDE is not None:
		E = float(E_OVERRIDE)
	else:
		E = float(ecirc + ENERGY_FRACTION * (E_ESCAPE - ecirc))

	if E <= ecirc:
		raise ValueError("Energy must be greater than E_circ(Lz) for a nontrivial SoS.")

	return E, Lz, ecirc, lzmax, rcirc


def zmax_at_R(pot, E, Lz, R):
	def f(z):
		return float(pot.potential(R, 0.0, z) - E + 0.5 * (Lz / R) ** 2)

	if f(0.0) > 0.0:
		return 0.0

	zhi = max(1.0, R)
	while f(zhi) < 0.0 and zhi < ZVC_BRACKET_MAX:
		zhi *= 2.0

	if zhi >= ZVC_BRACKET_MAX and f(zhi) < 0.0:
		raise RuntimeError("Could not bracket z-turning point. Increase ZVC_BRACKET_MAX.")

	return float(scipy.optimize.brentq(f, 0.0, zhi))


def make_zero_velocity_curves(pot, E, Lz, npts=N_GRID):
	rp, ra = pot.Rperiapo(E, Lz)
	if not np.isfinite(rp) or not np.isfinite(ra):
		raise RuntimeError("Could not compute (R_peri, R_apo) for this (E, Lz).")

	grid = np.linspace(0.0, 1.0, npts)
	grid = grid * grid * (3.0 - 2.0 * grid) * (ra - rp) + rp

	prmax = np.sqrt(
		np.maximum(
			0.0,
			2.0 * E - 2.0 * pot.potential(np.column_stack((grid, np.zeros_like(grid), np.zeros_like(grid))))
			- (Lz / grid) ** 2,
		)
	)

	zvals = np.zeros_like(grid)
	for i, rr in enumerate(grid[1:-1], start=1):
		zvals[i] = zmax_at_R(pot, E, Lz, rr)

	return float(rp), float(ra), grid, zvals, prmax


def orbit_time_for_seed(pot, E, seed_ic):
	tcirc = float(pot.Tcirc(E)) if np.isfinite(pot.Tcirc(E)) else 1.0
	if not np.isfinite(tcirc) or tcirc <= 0.0:
		tcirc = 1.0
	return ORBIT_TIME_MULTIPLIER * tcirc


@dataclass
class OrbitResult:
	ic: np.ndarray
	traj: np.ndarray
	traj_tail: np.ndarray
	sos_points: np.ndarray


def run_orbit(pot, E, Lz, R0, pR0):
	vz2 = 2.0 * (E - float(pot.potential(R0, 0.0, 0.0))) - (Lz / R0) ** 2 - pR0**2
	if vz2 <= 0.0:
		raise ValueError("Seed is outside the allowed zero-velocity region.")

	ic = np.array([R0, 0.0, 0.0, pR0, Lz / R0, np.sqrt(vz2)], dtype=float)
	tmax = orbit_time_for_seed(pot, E, ic)
	orbit = agama.orbit(ic=ic, potential=pot, time=tmax, trajsize=TRAJSIZE, dtype=object)

	traj = orbit(orbit)
	n_tail = max(2, int(ORBIT_TAIL_FRACTION * traj.shape[0]))
	traj_tail = traj[-n_tail:]

	timecross = orbit.z.roots()
	timecross = timecross[orbit.z(timecross, der=1) >= 0.0]
	if len(timecross) > 0:
		trajcross = orbit(timecross)
		rcross = np.hypot(trajcross[:, 0], trajcross[:, 1])
		prcross = (trajcross[:, 0] * trajcross[:, 3] + trajcross[:, 1] * trajcross[:, 4]) / rcross
		sos = np.column_stack((rcross, prcross))
	else:
		sos = np.empty((0, 2), dtype=float)

	return OrbitResult(ic=ic, traj=traj, traj_tail=traj_tail, sos_points=sos)


def plot_orbit_artists(ax_orbit, ax_sos, result, color):
	R_tail = np.hypot(result.traj_tail[:, 0], result.traj_tail[:, 1])
	z_tail = result.traj_tail[:, 2]

	orbit_line = ax_orbit.plot(R_tail, z_tail, color=color, lw=0.9, alpha=0.95)[0]
	orbit_seed = ax_orbit.plot([result.ic[0]], [result.ic[2]], marker="o", color=color, ms=4)[0]

	sos_scatter = None
	if result.sos_points.shape[0] > 0:
		sos_scatter = ax_sos.plot(
			result.sos_points[:, 0], result.sos_points[:, 1], "o", color=color, ms=1.7, mew=0, alpha=0.9
		)[0]
	sos_seed = ax_sos.plot([result.ic[0]], [result.ic[3]], marker="x", color=color, ms=6, mew=1.0)[0]

	return [orbit_line, orbit_seed, sos_scatter, sos_seed]


if __name__ == "__main__":
	pot = mcmillan_potential()
	E, Lz, Ecirc, Lzmax, Rcirc = choose_energy_and_lz(pot)
	Rp, Ra, Rgrid, Zmax, PRmax = make_zero_velocity_curves(pot, E, Lz)

	rng = np.random.default_rng(RNG_SEED)

	fig = plt.figure(figsize=(11.5, 5.5), dpi=130)
	ax_orbit = fig.add_axes((0.06, 0.12, 0.42, 0.82))
	ax_sos = fig.add_axes((0.54, 0.12, 0.40, 0.82))
	button_clear = Button(fig.add_axes((0.88, 0.965, 0.08, 0.028)), "clear")
	button_remove_last = Button(fig.add_axes((0.79, 0.965, 0.08, 0.028)), "rm last")
	button_remove_first = Button(fig.add_axes((0.70, 0.965, 0.08, 0.028)), "rm first")
	button_remove_second_last = Button(fig.add_axes((0.61, 0.965, 0.08, 0.028)), "rm 2nd")
	button_save_ics = Button(fig.add_axes((0.52, 0.965, 0.08, 0.028)), "save ICs")
	button_save_fig = Button(fig.add_axes((0.43, 0.965, 0.08, 0.028)), "save fig")
	status_text = fig.text(0.54, 0.035, "", fontsize=9, ha="left", va="bottom")
	orbit_artist_stack = []
	orbit_ic_stack = []

	def remove_orbit_artists(artists):
		for art in artists:
			if art is not None:
				art.remove()

	def init_axes(_event=None):
		orbit_artist_stack.clear()
		orbit_ic_stack.clear()
		ax_orbit.cla()
		ax_sos.cla()

		ax_orbit.set_title("Meridional plane orbit (R,z)")
		ax_orbit.set_xlabel("R")
		ax_orbit.set_ylabel("z")
		ax_orbit.set_aspect("equal")
		ax_orbit.set_xlim(max(0.0, 0.95 * Rp), 1.05 * Ra)
		zlim = 1.05 * max(np.max(Zmax), 1e-6)
		ax_orbit.set_ylim(-zlim, zlim)

		ax_sos.set_title("Surface of section: z=0, pz>0")
		ax_sos.set_xlabel("R")
		ax_sos.set_ylabel(r"$p_R$")
		ax_sos.set_xlim(max(0.0, 0.95 * Rp), 1.05 * Ra)
		ax_sos.set_ylim(-1.05 * np.max(PRmax), 1.05 * np.max(PRmax))

		R_orbit_poly = np.concatenate((Rgrid[:-1], Rgrid[::-1]))
		Z_orbit_poly = np.concatenate((Zmax[:-1], -Zmax[::-1]))
		ax_orbit.fill(R_orbit_poly, Z_orbit_poly, color="0.95", zorder=0)
		ax_orbit.plot(np.concatenate((Rgrid[:-1], Rgrid[::-1])), np.concatenate((Zmax[:-1], -Zmax[::-1])), color="k", lw=1.0)

		R_sos_poly = np.concatenate((Rgrid[:-1], Rgrid[::-1]))
		PR_sos_poly = np.concatenate((PRmax[:-1], -PRmax[::-1]))
		ax_sos.fill(R_sos_poly, PR_sos_poly, color="0.95", zorder=0)
		ax_sos.plot(Rgrid, PRmax, color="k", lw=1.0)
		ax_sos.plot(Rgrid, -PRmax, color="k", lw=1.0)

		ax_sos.text(
			0.02,
			0.98,
			"Right-click in this panel to launch an orbit",
			transform=ax_sos.transAxes,
			ha="left",
			va="top",
			fontsize=9,
		)
		ax_sos.text(
			0.02,
			0.90,
			f"E={E:.7g}, Lz={Lz:.7g}",
			transform=ax_sos.transAxes,
			ha="left",
			va="top",
			fontsize=9,
		)

		status_text.set_text("")
		fig.canvas.draw_idle()

	def remove_last_orbit(_event=None):
		if len(orbit_artist_stack) == 0:
			status_text.set_text("No orbit to remove.")
			fig.canvas.draw_idle()
			return

		remove_orbit_artists(orbit_artist_stack.pop())
		orbit_ic_stack.pop()
		status_text.set_text(f"Removed latest orbit. Remaining={len(orbit_artist_stack)}")
		fig.canvas.draw_idle()

	def remove_first_orbit(_event=None):
		if len(orbit_artist_stack) == 0:
			status_text.set_text("No orbit to remove.")
			fig.canvas.draw_idle()
			return

		remove_orbit_artists(orbit_artist_stack.pop(0))
		orbit_ic_stack.pop(0)
		status_text.set_text(f"Removed first orbit. Remaining={len(orbit_artist_stack)}")
		fig.canvas.draw_idle()

	def remove_second_last_orbit(_event=None):
		if len(orbit_artist_stack) < 2:
			status_text.set_text("Need at least two orbits to remove the second-to-last one.")
			fig.canvas.draw_idle()
			return

		idx = len(orbit_artist_stack) - 2
		remove_orbit_artists(orbit_artist_stack.pop(idx))
		orbit_ic_stack.pop(idx)
		status_text.set_text(f"Removed second-to-last orbit. Remaining={len(orbit_artist_stack)}")
		fig.canvas.draw_idle()

	def save_initial_conditions(_event=None):
		if len(orbit_ic_stack) == 0:
			status_text.set_text("No initial conditions to save.")
			fig.canvas.draw_idle()
			return

		# Orbit seeds are stored as Cartesian phase-space vectors:
		# [x, y, z, vx, vy, vz].
		arr = np.array(orbit_ic_stack, dtype=float)
		np.savetxt(
			SAVE_IC_FILENAME,
			arr,
			fmt="%.16e",
			header=SAVE_IC_HEADER,
			comments="",
		)
		status_text.set_text(f"Saved {arr.shape[0]} Cartesian initial conditions to {SAVE_IC_FILENAME}")
		print(f"Saved {arr.shape[0]} Cartesian initial conditions to {SAVE_IC_FILENAME}")
		fig.canvas.draw_idle()

	def save_figure(_event=None):
		fig.savefig(SAVE_FIG_FILENAME, dpi=300)
		status_text.set_text(f"Saved figure to {SAVE_FIG_FILENAME}")
		print(f"Saved figure to {SAVE_FIG_FILENAME}")
		fig.canvas.draw_idle()

	def add_orbit_from_seed(R0, pR0):
		if R0 <= 0.0:
			status_text.set_text("Invalid seed: R must be positive.")
			fig.canvas.draw_idle()
			return

		try:
			result = run_orbit(pot, E, Lz, R0, pR0)
		except ValueError as exc:
			status_text.set_text(str(exc))
			fig.canvas.draw_idle()
			return

		color = rng.uniform(0.15, 0.85, size=3)
		orbit_artist_stack.append(plot_orbit_artists(ax_orbit, ax_sos, result, color=color))
		orbit_ic_stack.append(result.ic.copy())

		status_text.set_text(f"Ncross={result.sos_points.shape[0]} | seed=(R,pR)=({R0:.5g},{pR0:.5g})")
		print(
			"seed (R,pR,pz) = ({:.6f}, {:.6f}, {:.6f}) | crossings={}".format(
				result.ic[0], result.ic[3], result.ic[5], result.sos_points.shape[0]
			)
		)
		fig.canvas.draw_idle()

	def on_click(event):
		if event.inaxes is not ax_sos:
			return
		if event.button != 3:
			return
		if event.xdata is None or event.ydata is None:
			return
		add_orbit_from_seed(float(event.xdata), float(event.ydata))

	def on_key(event):
		key = "" if event.key is None else event.key.lower()
		if key in ("1",):
			remove_last_orbit()
		elif key in ("2",):
			remove_second_last_orbit()
		elif key in ("f",):
			remove_first_orbit()
		elif key in ("c",):
			init_axes()
		elif key in ("s",):
			save_initial_conditions()

	fig.canvas.mpl_connect("button_press_event", on_click)
	fig.canvas.mpl_connect("key_press_event", on_key)
	button_clear.on_clicked(init_axes)
	button_remove_last.on_clicked(remove_last_orbit)
	button_remove_first.on_clicked(remove_first_orbit)
	button_remove_second_last.on_clicked(remove_second_last_orbit)
	button_save_ics.on_clicked(save_initial_conditions)
	button_save_fig.on_clicked(save_figure)
	init_axes()

	print("Interactive meridional SoS explorer ready.")
	print("Right-click in the right panel (R,p_R) to launch an orbit at fixed (E,Lz).")
	print("Hotkeys: 1=remove last, 2=remove second-to-last, f=remove first, s=save ICs, c=clear all.")
	print(f"Using {POTENTIAL_FILENAME}, E={E:.7g}, Lz={Lz:.7g}, E_circ={Ecirc:.7g}, Lzmax={Lzmax:.7g}")
	plt.show()
