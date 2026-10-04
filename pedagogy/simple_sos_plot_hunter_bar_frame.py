import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import agama

import barframe
import hunter2024mwmodel

agama.setUnits(length=1, mass=1, velocity=1)


def accel_rotating_frame(x, y, vx, vy, pot, omega):
    a = pot.force(np.array([[x, y, 0.0]], dtype=np.float64))[0]
    ax = a[0] + omega**2 * x + 2.0 * omega * vy
    ay = a[1] + omega**2 * y - 2.0 * omega * vx
    return np.array([ax, ay], dtype=np.float64)


def kick_half(state, dt, pot, omega):
    x, y, vx, vy = state
    ax, ay = accel_rotating_frame(x, y, vx, vy, pot, omega)
    vx += 0.5 * dt * ax
    vy += 0.5 * dt * ay
    return np.array([x, y, vx, vy], dtype=np.float64)


def drift(state, dt):
    x, y, vx, vy = state
    x += dt * vx
    y += dt * vy
    return np.array([x, y, vx, vy], dtype=np.float64)


def strang_rotating_frame(state, t_span, dt, pot, omega):
    t0, tf = t_span
    nsteps = int(np.ceil((tf - t0) / dt))
    dt = (tf - t0) / nsteps

    times = t0 + dt * np.arange(nsteps + 1)
    orbit = np.empty((nsteps + 1, 4), dtype=np.float64)
    orbit[0] = state

    for i in range(nsteps):
        tmp = kick_half(orbit[i], dt, pot, omega)
        tmp = drift(tmp, dt)
        tmp = kick_half(tmp, dt, pot, omega)
        orbit[i + 1] = tmp

    return times, orbit


def jacobi_energy_rotating_frame(state, pot, omega):
    x = state[:, 0]
    y = state[:, 1]
    vx = state[:, 2]
    vy = state[:, 3]
    kinetic = 0.5 * (vx**2 + vy**2)
    effective_potential = barframe.pot_eff(
        pot, omega, np.column_stack((x, y, np.zeros_like(x)))
    )
    return kinetic + effective_potential


def sos_points_y_crossings_upward(orbit):
    """Return (x, vx) at upward y=0 crossings using linear interpolation."""
    x0 = orbit[:-1, 0]
    y0 = orbit[:-1, 1]
    vx0 = orbit[:-1, 2]
    x1 = orbit[1:, 0]
    y1 = orbit[1:, 1]
    vx1 = orbit[1:, 2]

    mask = (y0 <= 0.0) & (y1 > 0.0)
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float64)

    dy = y1[mask] - y0[mask]
    alpha = -y0[mask] / dy
    xcross = x0[mask] + alpha * (x1[mask] - x0[mask])
    vxcross = vx0[mask] + alpha * (vx1[mask] - vx0[mask])
    return np.column_stack((xcross, vxcross))


def vy_from_energy_on_section(x, vx, pot, omega, ej):
    """Solve vy from y=0 Jacobi section condition for given (x, vx)."""
    peff = barframe.pot_eff(pot, omega, np.array([x, 0.0, 0.0], dtype=np.float64))
    vy2 = 2.0 * (ej - peff) - vx**2
    if vy2 <= 0.0:
        return np.nan
    return float(np.sqrt(vy2))


def build_zero_velocity_curve(pot, omega, ej, tolerance=1e-3, npts=501):
    xmax = float(barframe.xmaximum_within_bar(pot, omega, ej))
    xs = np.linspace(-(1.0 - tolerance) * xmax, (1.0 - tolerance) * xmax, npts)
    vxs = barframe.vx_zero_velocity_curve(xs, pot, omega, ej)
    xs_poly = np.concatenate((xs, xs[::-1], [xs[0]]))
    vxs_poly = np.concatenate((vxs, -vxs[::-1], [vxs[0]]))
    return xmax, xs, vxs, xs_poly, vxs_poly


# Configuration for interactivity and integration speed/accuracy balance.
omega = -37.5
time_span = (0.0, 3.0)
dt = 2e-4
jacobi_warn_threshold = 1e-4
rng = np.random.default_rng(seed=13)


pot = hunter2024mwmodel.mw_model()
xL1 = barframe.xL1(pot, omega)
EL1 = barframe.pot_eff(pot, omega, np.array([xL1, 0.0, 0.0]))
E0 = barframe.pot_eff(pot, omega, np.array([0.0, 0.0, 0.0]))

# Set Jacobi energy here.
Ej = (49.0 / 50.0) * (EL1 - E0) + E0
xmax, xs, vxs, xs_poly, vxs_poly = build_zero_velocity_curve(pot, omega, Ej)


fig = plt.figure(figsize=(11.0, 5.3), dpi=130)
ax_orbit = fig.add_axes((0.07, 0.13, 0.42, 0.80))
ax_sos = fig.add_axes((0.55, 0.13, 0.40, 0.80))
button_clear = Button(fig.add_axes((0.86, 0.92, 0.10, 0.06)), "clear")

status_text = fig.text(0.55, 0.04, "", fontsize=9, ha="left", va="bottom")


def init_axes(_event=None):
    ax_orbit.cla()
    ax_sos.cla()

    ax_orbit.set_title("Orbit in bar frame")
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
    ax_sos.plot(xs, vxs, color="black", lw=1.0)
    ax_sos.plot(xs, -vxs, color="black", lw=1.0)
    ax_sos.fill(xs_poly, vxs_poly, color="0.95", zorder=0)

    ax_sos.text(
        0.02,
        0.98,
        "Click in (x, vx) panel to launch orbit",
        transform=ax_sos.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax_sos.text(
        0.02,
        0.90,
        f"Ej = {Ej:.6g}",
        transform=ax_sos.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    status_text.set_text("")
    fig.canvas.draw_idle()


def run_orbit_from_section_click(x, vx):
    vy = vy_from_energy_on_section(x=x, vx=vx, pot=pot, omega=omega, ej=Ej)
    if not np.isfinite(vy):
        status_text.set_text("Invalid click: outside allowed zero-velocity region.")
        fig.canvas.draw_idle()
        return

    state0 = np.array([x, 0.0, vx, vy], dtype=np.float64)
    _times, orbit = strang_rotating_frame(state0, t_span=time_span, dt=dt, pot=pot, omega=omega)

    ej_orbit = jacobi_energy_rotating_frame(orbit, pot, omega)
    denom = max(abs(Ej), 1e-12)
    rel_err = np.abs((ej_orbit - Ej) / denom)
    max_rel_err = float(np.max(rel_err))

    sos_points = sos_points_y_crossings_upward(orbit)
    color = rng.uniform(0.15, 0.85, size=3)

    ax_orbit.plot(orbit[:, 0], orbit[:, 1], color=color, lw=0.9, alpha=0.95)
    ax_orbit.plot([state0[0]], [state0[1]], marker="o", color=color, ms=4)

    if sos_points.shape[0] > 0:
        ax_sos.plot(sos_points[:, 0], sos_points[:, 1], "o", color=color, ms=1.7, mew=0, alpha=0.9)
    ax_sos.plot([state0[0]], [state0[2]], marker="x", color=color, ms=6, mew=1.0)

    status = (
        f"Ncross={sos_points.shape[0]}   "
        f"| max rel Jacobi err={max_rel_err:.3e}"
    )
    if max_rel_err > jacobi_warn_threshold:
        status += "   [warning: decrease dt for tighter conservation]"
    status_text.set_text(status)

    print(
        "seed (x, vx, vy)=({:.6f}, {:.6f}, {:.6f}) | crossings={} | max rel Jacobi err={:.3e}".format(
            state0[0], state0[2], state0[3], sos_points.shape[0], max_rel_err
        )
    )
    fig.canvas.draw_idle()


def add_point(event):
    if event.inaxes is not ax_sos:
        return
    if event.xdata is None or event.ydata is None:
        return
    run_orbit_from_section_click(float(event.xdata), float(event.ydata))


fig.canvas.mpl_connect("button_press_event", add_point)
button_clear.on_clicked(init_axes)
init_axes()

print("Interactive bar-frame SOS ready.")
print("Click in the right panel (x, vx) to integrate an orbit at fixed Jacobi energy.")
plt.show()