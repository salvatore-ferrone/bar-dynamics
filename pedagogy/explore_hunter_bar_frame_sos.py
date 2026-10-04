import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import agama

import barframe
import hunter2024mwmodel


agama.setUnits(length=1, mass=1, velocity=1)


def jacobi_energy(traj, pot, omega):
	"""Jacobi constant evaluated from rotating-frame states [x,y,z,vx,vy,vz]."""
	x = traj[:, 0]
	y = traj[:, 1]
	z = traj[:, 2]
	vx = traj[:, 3]
	vy = traj[:, 4]
	vz = traj[:, 5]
	kinetic = 0.5 * (vx**2 + vy**2 + vz**2)
	peff = barframe.pot_eff(pot, omega, np.column_stack((x, y, z)))
	return kinetic + peff


def vy_from_section_energy(x, vx, ej, pot, omega):
	"""Given (x,vx) on y=z=vz=0 section, solve for positive vy at fixed Ej."""
	peff = barframe.pot_eff(pot, omega, np.array([x, 0.0, 0.0], dtype=np.float64))
	vy2 = 2.0 * (ej - peff) - vx**2
	if vy2 <= 0.0:
		return np.nan

	return float(np.sqrt(vy2))


def sos_points_from_traj(traj, ej, pot, omega):
	"""Extract (x,vx) for upward y=0 crossings from sampled trajectory.

	A small Jacobi-consistent correction keeps points inside the allowed
	zero-velocity region when interpolation noise pushes them slightly outside.
	"""
	y0 = traj[:-1, 1]
	y1 = traj[1:, 1]
	mask = (y0 <= 0.0) & (y1 > 0.0)
	if not np.any(mask):
		return np.empty((0, 2), dtype=np.float64)

	t0 = traj[:-1][mask]
	t1 = traj[1:][mask]
	alpha = -t0[:, 1] / (t1[:, 1] - t0[:, 1])
	xcross = t0[:, 0] + alpha * (t1[:, 0] - t0[:, 0])
	vxcross = t0[:, 3] + alpha * (t1[:, 3] - t0[:, 3])
	vycross = t0[:, 4] + alpha * (t1[:, 4] - t0[:, 4])

	keep_upward = vycross > 0.0
	if not np.any(keep_upward):
		return np.empty((0, 2), dtype=np.float64)

	xcross = xcross[keep_upward]
	vxcross = vxcross[keep_upward]

	peff = barframe.pot_eff(
		pot,
		omega,
		np.column_stack((xcross, np.zeros_like(xcross), np.zeros_like(xcross))),
	)
	vx2_allowed = 2.0 * np.maximum(ej - peff, 0.0)
	vxmax = np.sqrt(vx2_allowed)

	# Keep sign but clip tiny overshoots to the allowed boundary.
	vxcross = np.sign(vxcross) * np.minimum(np.abs(vxcross), vxmax)
	return np.column_stack((xcross, vxcross))


def make_zvc(ej, pot, omega, tolerance=1e-3, npts=601):
	xmax = float(barframe.xmaximum_within_bar(pot, omega, ej))
	xs = np.linspace(-(1.0 - tolerance) * xmax, (1.0 - tolerance) * xmax, npts)
	vxs = barframe.vx_zero_velocity_curve(xs, pot, omega, ej)
	xs_poly = np.concatenate((xs, xs[::-1], [xs[0]]))
	vxs_poly = np.concatenate((vxs, -vxs[::-1], [vxs[0]]))
	return xmax, xs, vxs, xs_poly, vxs_poly


# ----------------------- user-tunable parameters -----------------------
omega = -37.5
integration_time = 10.0
n_samples = 12000
jacobi_warn_threshold = 5e-8
orbit_tail_fraction = 0.10

# Pick Jacobi energy here.
ej_fraction = 49/50
# ----------------------------------------------------------------------


pot = hunter2024mwmodel.mw_model()
x_l1 = barframe.xL1(pot, omega)
e_l1 = barframe.pot_eff(pot, omega, np.array([x_l1, 0.0, 0.0]))
e_0 = barframe.pot_eff(pot, omega, np.array([0.0, 0.0, 0.0]))
Ej = ej_fraction * (e_l1 - e_0) + e_0

xmax, xs, vxs, xs_poly, vxs_poly = make_zvc(Ej, pot, omega)
rng = np.random.default_rng(17)


fig = plt.figure(figsize=(11.5, 5.5), dpi=130)
ax_orbit = fig.add_axes((0.06, 0.12, 0.42, 0.82))
ax_sos = fig.add_axes((0.54, 0.12, 0.40, 0.82))
button_clear = Button(fig.add_axes((0.88, 0.965, 0.08, 0.028)), "clear")
button_remove_last = Button(fig.add_axes((0.79, 0.965, 0.08, 0.028)), "rm last")
button_remove_first = Button(fig.add_axes((0.70, 0.965, 0.08, 0.028)), "rm first")
status_text = fig.text(0.54, 0.035, "", fontsize=9, ha="left", va="bottom")
orbit_artist_stack = []


def init_axes(_event=None):
	orbit_artist_stack.clear()
	ax_orbit.cla()
	ax_sos.cla()

	ax_orbit.set_title("Orbit in bar frame (x,y)")
	ax_orbit.set_xlabel("x")
	ax_orbit.set_ylabel("y")
	ax_orbit.set_aspect("equal")
	ax_orbit.set_xlim(-1.05 * xmax, 1.05 * xmax)
	ax_orbit.set_ylim(-1.05 * xmax, 1.05 * xmax)

	ax_sos.set_title("Surface of section: y=0, vy>0")
	ax_sos.set_xlabel("x")
	ax_sos.set_ylabel("vx")
	ax_sos.set_xlim(-1.05 * xmax, 1.05 * xmax)
	ax_sos.set_ylim(-1.05 * np.max(vxs), 1.05 * np.max(vxs))
	ax_sos.fill(xs_poly, vxs_poly, color="0.95", zorder=0)
	ax_sos.plot(xs, vxs, color="k", lw=1.0)
	ax_sos.plot(xs, -vxs, color="k", lw=1.0)

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
		f"Ej={Ej:.7g}",
		transform=ax_sos.transAxes,
		ha="left",
		va="top",
		fontsize=9,
	)

	status_text.set_text("")
	fig.canvas.draw_idle()


def remove_orbit_artists(orbit_artists):
	for artist in orbit_artists:
		if artist is not None:
			artist.remove()


def remove_last_orbit(_event=None):
	if len(orbit_artist_stack) == 0:
		status_text.set_text("No orbit to remove.")
		fig.canvas.draw_idle()
		return

	remove_orbit_artists(orbit_artist_stack.pop())
	status_text.set_text(f"Removed latest orbit. Remaining={len(orbit_artist_stack)}")
	fig.canvas.draw_idle()


def remove_first_orbit(_event=None):
	if len(orbit_artist_stack) == 0:
		status_text.set_text("No orbit to remove.")
		fig.canvas.draw_idle()
		return

	remove_orbit_artists(orbit_artist_stack.pop(0))
	status_text.set_text(f"Removed first orbit. Remaining={len(orbit_artist_stack)}")
	fig.canvas.draw_idle()


def integrate_with_agama(state0):
	"""Integrate one rotating-frame orbit using Agama's native integrator."""
	orbit_spline = agama.orbit(
		ic=state0,
		potential=pot,
		Omega=omega,
		time=integration_time,
		dtype=object,
	)
	times = np.linspace(0.0, integration_time, n_samples)
	traj = orbit_spline(times)
	return times, traj


def add_orbit_from_section(x, vx):
	vy = vy_from_section_energy(x, vx, Ej, pot, omega)
	if not np.isfinite(vy):
		status_text.set_text("Invalid click: outside the allowed zero-velocity region.")
		fig.canvas.draw_idle()
		return

	state0 = np.array([x, 0.0, 0.0, vx, vy, 0.0], dtype=np.float64)
	_times, traj = integrate_with_agama(state0)
	n_tail = max(2, int(orbit_tail_fraction * traj.shape[0]))
	traj_plot = traj[-n_tail:]

	ej_traj = jacobi_energy(traj, pot, omega)
	denom = max(abs(Ej), 1e-12)
	rel_err = np.abs((ej_traj - Ej) / denom)
	max_rel_err = float(np.max(rel_err))

	sos = sos_points_from_traj(traj, Ej, pot, omega)
	color = rng.uniform(0.15, 0.85, size=3)

	orbit_line = ax_orbit.plot(
		traj_plot[:, 0], traj_plot[:, 1], color=color, lw=0.9, alpha=0.95
	)[0]
	orbit_seed_marker = ax_orbit.plot(
		[state0[0]], [state0[1]], marker="o", color=color, ms=4
	)[0]

	sos_points_artist = None
	if sos.shape[0] > 0:
		sos_points_artist = ax_sos.plot(
			sos[:, 0], sos[:, 1], "o", color=color, ms=1.7, mew=0, alpha=0.9
		)[0]
	sos_seed_marker = ax_sos.plot(
		[state0[0]], [state0[3]], marker="x", color=color, ms=6, mew=1.0
	)[0]

	orbit_artist_stack.append(
		[
			orbit_line,
			orbit_seed_marker,
			sos_points_artist,
			sos_seed_marker,
		]
	)

	msg = f"Ncross={sos.shape[0]} | max rel Jacobi err={max_rel_err:.3e}"
	if max_rel_err > jacobi_warn_threshold:
		msg += " [warning: raise n_samples or reduce step via shorter integration chunks]"
	status_text.set_text(msg)
	print(
		"seed (x,vx,vy)=({:.6f},{:.6f},{:.6f}) | crossings={} | max rel Jacobi err={:.3e}".format(
			state0[0], state0[3], state0[4], sos.shape[0], max_rel_err
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
	add_orbit_from_section(float(event.xdata), float(event.ydata))


def on_key(event):
	key = "" if event.key is None else event.key.lower()
	if key in ("backspace", "u"):
		remove_last_orbit()
	elif key in ("1",):
		remove_first_orbit()
	elif key in ("c",):
		init_axes()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
button_clear.on_clicked(init_axes)
button_remove_last.on_clicked(remove_last_orbit)
button_remove_first.on_clicked(remove_first_orbit)
init_axes()

print("Interactive Hunter+bar-frame SOS explorer ready.")
print("Right-click in the right panel (x,vx) to launch an orbit at fixed Ej.")
print("Hotkeys: u/backspace=remove last, 1=remove first, c=clear all.")
plt.show()
