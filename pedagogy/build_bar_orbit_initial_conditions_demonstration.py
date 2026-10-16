import os
import agama, numpy, matplotlib, matplotlib.pyplot as plt
import hunter2024mwmodel as h2024
import barframe
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


def build_total_potential():
    bar1 = h2024.coleman2020(**h2024.coleman_components)
    bar2 = h2024.portail_component(**h2024.portail_bar_1)
    bar3 = h2024.portail_component(**h2024.portail_bar_2)
    cyl = dict(type='CylSpline', symmetry='triaxial', mmax=8, gridSizeR=40, gridSizeZ=40, Rmin=0.001, Rmax=20, Zmin=0.001, Zmax=20)
    pbar1 = agama.Potential(density=bar1, **cyl)
    pbar2 = agama.Potential(density=bar2, **cyl)
    pbar3 = agama.Potential(density=bar3, **cyl)
    rest = [agama.Potential(**c) for c in (
        h2024.supermassiveblackhole, h2024.nsc, h2024.NSD1, h2024.NSD2,
        h2024.DISK1, h2024.DISK2, h2024.GDISK1, h2024.GDISK2, h2024.EINASTO)]
    return agama.Potential(pbar1, pbar2, pbar3, *rest)


def make_frame_saver(figbasename):
    os.makedirs(os.path.dirname(figbasename), exist_ok=True)
    counter = [0]
    def save(fig):
        fig.tight_layout()
        fig.savefig(figbasename.format(counter[0]), dpi=300)
        counter[0] += 1
    return save


def draw_demonstration(pot_total, omega, seed, figbasename, npoints=201, threshold=1e-3, weighting="phase_space"):
    # weighting: "none", "volume" or "phase_space" (see barframe.sample_position_within_bar)
    rng = numpy.random.default_rng(seed)
    save = make_frame_saver(figbasename)

    # one sample, built step by step with the barframe functions
    x_l1, e_l1 = barframe.lagrange_l1_energy(pot_total, omega)
    y_l4 = barframe.yL4(pot_total, omega)
    e_0 = barframe.pot_eff(pot_total, omega, numpy.array([0.0, 0.0, 0.0]))
    e_l4 = barframe.pot_eff(pot_total, omega, numpy.array([0, y_l4, 0]))

    x_sample, y_sample, z_sample = barframe.sample_position_within_bar(pot_total, omega, rng, x_l1, e_l1, weighting)
    y_max = barframe.ymax_within_bar(pot_total, omega, x_sample, e_l1, x_l1 / 2)
    availablespeed = barframe.available_speed(pot_total, omega, [x_sample, y_sample, z_sample], e_l1)
    _, v_sample, theta, phi = barframe.sample_velocity_isotropic(availablespeed, rng)

    # zero velocity curve in the yz-plane at x_sample
    y_zvc = numpy.linspace(-(1 - threshold) * y_max, (1 - threshold) * y_max, npoints)
    z_zvc = numpy.array([barframe.zmax_within_bar(pot_total, omega, x_sample, yy, e_l1, y_max) for yy in y_zvc])
    zvc_y = numpy.concatenate((y_zvc, y_zvc[::-1], [y_zvc[0]]))
    zvc_z = numpy.concatenate((z_zvc, -z_zvc[::-1], [z_zvc[0]]))

    # contours of the effective potential in the x-y plane
    xfactor, yfactor = 4, 1.2
    xs = numpy.linspace(-xfactor * x_l1, xfactor * x_l1, npoints)
    ys = numpy.linspace(-yfactor * x_l1, yfactor * x_l1, npoints)
    XS, YS = numpy.meshgrid(xs, ys)
    pos = numpy.column_stack((XS.flatten(), YS.flatten(), 0 * YS.flatten()))
    pot_eff = numpy.reshape(barframe.pot_eff(pot_total, omega, pos), XS.shape)
    contours = numpy.linspace(pot_eff.min(), pot_eff.max(), 10)
    contours = numpy.sort(numpy.concatenate((contours, numpy.linspace(e_l1 * (1 + .4e-3), e_l4, 5))))

    # slices used to show the x and y sampling
    phi_x = barframe.pot_eff(pot_total, omega, numpy.column_stack([xs, xs * 0, xs * 0]))
    phi_x_sample = barframe.pot_eff(pot_total, omega, numpy.array([x_sample, 0, 0]))
    phi_0 = barframe.pot_eff(pot_total, omega, numpy.array([0.1, 0.1, 0.1]))
    phi_y_at_x = barframe.pot_eff(pot_total, omega, numpy.column_stack((x_sample * numpy.ones_like(ys), ys, 0 * ys)))
    phi_y_max_at_x = barframe.pot_eff(pot_total, omega, numpy.array([x_sample, y_max, 0]))
    phi_y_at_sample = barframe.pot_eff(pot_total, omega, numpy.array([x_sample, y_sample, 0]))

    nrow, ncol = 3, 2
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(nrow, ncol, figure=fig)
    axis = fig.add_subplot(gs[0, :])
    axes = [[fig.add_subplot(gs[i, j]) for j in range(ncol)] for i in range(1, nrow)]
    # the last panel is split later into the speed bar and the mollweide
    axes[1][1].remove()
    contour1 = axis.contour(XS, YS, pot_eff, levels=contours, colors="k", linestyles="-")
    axis.set(aspect="equal", xlabel="x", ylabel="y")
    axes[0][0].set(xlabel="x", ylabel=r"$\Phi_{\mathrm{eff}}(x,0,0)$", yticks=[], xlim=[xs.min(), xs.max()], ylim=[phi_0, (1 - 3e-2) * phi_x.max()])
    axes[0][1].set(ylabel=r"$\Phi_{\mathrm{eff}}(x_s,y,0)$", yticks=[], xlabel="y", xlim=[ys.min(), ys.max()], ylim=[phi_y_at_x.min(), (1 - 1e-2) * phi_y_at_x.max()])
    axes[1][0].set(xlabel="y", ylabel="z")
    # draw once so the empty layout is frame 0, then hide the ZVC and contours until their frame
    obj, = axes[1][0].plot(zvc_y, zvc_z, color="k")
    fig.tight_layout()
    obj.remove()
    contour1.remove()
    save(fig)

    axis.contour(XS, YS, pot_eff, levels=contours, colors="k", linestyles="-")
    axis.set(title=r"Milky-Way $\Phi_{\mathrm{eff}}$ Hunter et al. (2024)")
    save(fig)

    axes[0][0].plot(xs, phi_x, color="k")
    save(fig)

    # the box that x_sample is drawn from
    axes[0][0].hlines(e_l1, -x_l1, x_l1, color="k", linewidth=1)
    axes[0][0].vlines([-x_l1, x_l1], e_0, e_l1, color="k", linewidth=1)
    save(fig)

    axes[0][0].scatter(x_sample, phi_x_sample, color="r", zorder=10)
    axes[0][0].hlines(phi_x_sample, -numpy.abs(x_sample), numpy.abs(x_sample), color="k")
    save(fig)

    mydot = axis.scatter(x_sample, 0, color="r", zorder=10)
    save(fig)

    axes[0][1].plot(ys, phi_y_at_x, color="k")
    save(fig)

    axes[0][1].vlines([-y_max, y_max], phi_y_at_x.min(), phi_y_max_at_x, color="k")
    axes[0][1].hlines(e_l1, -y_max, y_max, color="k")
    save(fig)

    axes[0][1].scatter([y_sample], [phi_y_at_sample], zorder=10, color="r")
    save(fig)

    mydot.remove()
    axis.scatter([x_sample], [y_sample], zorder=10, color="r")
    save(fig)

    axes[1][0].set(title=r"Zero Velocity Curve at $x_s$", aspect="equal")
    axes[1][0].plot(zvc_y, zvc_z, color="k")
    save(fig)

    axes[1][0].scatter(y_sample, z_sample, color="r")
    save(fig)

    # nested grid: mollweide (2/3) beside a skinny speed bar (1/3)
    gs_vel = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 1], width_ratios=[2, 1], wspace=-0.35)
    axis_speed = fig.add_subplot(gs_vel[0, 1])
    axis_speed.axis("off")
    eventplot, = axis_speed.eventplot([0, availablespeed], orientation="vertical", color="k")
    save(fig)

    axis_speed.text(1, (1 + 5e-2) * availablespeed, s=r"$v_{\mathrm{available}}$", va="bottom", ha="center")
    save(fig)

    axis_speed.text(1, -(1e-1), s=r"$v=0$", va="top", ha="center")
    save(fig)

    eventplot.remove()
    axis_speed.eventplot([0, v_sample, availablespeed], orientation="vertical", color="k")
    axis_speed.text(1.5, v_sample, s=r"$v_{\mathrm{s}}$", va="center")
    save(fig)

    axis_angle = fig.add_subplot(gs_vel[0, 0], projection="mollweide")
    axis_angle.grid(True)
    axis_angle.set_xticklabels([])
    axis_angle.set_yticklabels([])
    save(fig)

    axis_angle.set(title="Isotropic angle")
    save(fig)

    # mollweide latitude is pi/2 - theta
    axis_angle.plot(phi, numpy.pi / 2 - theta, "o", color="k")
    save(fig)
    return fig


if __name__ == "__main__":
    omega = -37.5
    pot_total = build_total_potential()
    draw_demonstration(pot_total, omega, seed=11, figbasename="plots/bar_init_conds/bar_init_conds_fig_frame_{:03d}.png")
