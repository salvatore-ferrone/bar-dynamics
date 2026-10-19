'''
helper functions for the bar frame 
'''

import numpy as np 
from scipy import optimize 

def pot_eff(pot,omega,pos):
    potential=pot.potential(pos)
    if pos.ndim==1:
        CENTRIFUGAL = (1/2)*omega**2 * (pos[0]**2 + pos[1]**2)
    else: 
        CENTRIFUGAL = (1/2)*omega**2 * (pos[:,0]**2 + pos[:,1]**2)
    return potential-CENTRIFUGAL

def force_eff(pot,omega,pos):
    f = pot.force(pos)
    if pos.ndim==1:
        f[:, 0] += omega**2 * pos[0]
        f[:, 1] += omega**2 * pos[1]
    else:
        f[:, 0] += omega**2 * pos[:, 0]
        f[:, 1] += omega**2 * pos[:, 1]
    return f

# the quick anonymous functions 
fx = lambda x, pot, omega: force_eff(pot, omega, np.array([[x, 0., 0.]]))[0, 0]
fy = lambda y, pot, omega: force_eff(pot, omega, np.array([[0., y, 0.]]))[0, 1]
x_max_root_within_bar = lambda x, pot, omega, Ej: pot_eff(pot,omega, np.array([x,0,0])) - Ej

# get the locations of the lagrange points 

def xL1(pot,omega,xmin=0.1, xmax=20):
    return float(optimize.brentq(fx, xmin, xmax, args=(pot, omega)))      # bracket needs a sign change

def yL4(pot,omega,ymin=0.1, ymax=20):
    return float(optimize.brentq(fy, ymin, ymax, args=(pot, omega)))

def xmaximum_within_bar(pot,omega,Ej, xmin=0.1, xmax=20, tolerance = 1e-3):
    """
    Get the maxmium extent allowed for an interior bar orbit 
    Ej must be less than that energy to work. Otherwise nan is return
    """
    myxL1 = xL1(pot,omega, xmin=xmin, xmax=xmax)
    EL1 = pot_eff(pot,omega,pos=np.array([myxL1,0,0]))
    if (Ej > EL1) :
        print("warning, Given jacobi energy allows particle to escape the bar")
        return np.nan
    # find the root within the origin and the lagrange point 
    return float(optimize.brenth(x_max_root_within_bar, xmin, (1-tolerance)*myxL1, args=(pot,omega,Ej)))

def vx_zero_velocity_curve(x,pot,omega,Ej):
    """
    zero-velocity curve is really a mis-nomer
    Becaues, it stems from looking at the maximum allowed extent in the x-y plane
    however, now we're looking at the x-vx plane. So, we're looking at the fatherest allowed 
    extent from the origin within this plane. The name carried over from x-y plane. 
    """
    pos = np.array([x,np.zeros_like(x),np.zeros_like(x)]).T
    kinetic_energy = Ej - pot_eff(pot,omega,pos)
    return np.sqrt(2*np.maximum(kinetic_energy,0.0))

def xy_plane_zero_velocity_curve(pot,omega,Ej,nangles=360,nr=2000):
    '''
    Given a jacobi energy, find the zero velocity curve in the x-y plane. 
    Returns closed-curve arrays (x, y) of the boundary connected to the origin, found as the 
    first crossing of pot_eff = Ej along rays from the origin. Needs Ej <= E_L1, else nan arrays.
    '''
    xl1, EL1 = lagrange_l1_energy(pot, omega)
    phi = np.linspace(0, 2*np.pi, nangles, endpoint=False)
    if Ej > EL1:
        print("warning, Given jacobi energy allows particle to escape the bar")
        return np.full(nangles+1, np.nan), np.full(nangles+1, np.nan)
    # slightly beyond L1 so the Ej = E_L1 curve is still bracketed along the bar axis
    r = np.linspace(1e-3, 1.05*xl1, nr)
    pos = np.zeros((nangles, nr, 3))
    pos[..., 0] = np.cos(phi)[:, None] * r
    pos[..., 1] = np.sin(phi)[:, None] * r
    g = pot_eff(pot, omega, pos.reshape(-1, 3)).reshape(nangles, nr) - Ej
    # first index where the ray leaves the allowed region (g > 0 means kinetic energy < 0)
    outside = g > 0
    i = np.where(outside.any(axis=1), outside.argmax(axis=1), nr - 1)
    i = np.maximum(i, 1)
    rows = np.arange(nangles)
    g0, g1 = g[rows, i-1], g[rows, i]
    # linear interpolation of the root between the bracketing grid points
    frac = np.clip(-g0 / np.where(g1 == g0, 1, g1 - g0), 0, 1)
    rb = r[i-1] + frac * (r[i] - r[i-1])
    x, y = rb*np.cos(phi), rb*np.sin(phi)
    return np.append(x, x[0]), np.append(y, y[0])

def sample_initial_conditions_allowed_within_bar(potential,omega,Ej,norbits=50,seed=10,xmax_kwargs={"xmin": 0.1, "xmax": 20, "tolerance": 1e-3},):
    # Get the maximum extent in x, when kinetic energy is zero 
    xmax = xmaximum_within_bar(pot=potential,omega=omega,Ej=Ej,**xmax_kwargs)
    # pick a random x between these two 
    rng = np.random.default_rng(seed=seed)
    k = 0 
    initial_conditions = np.zeros((norbits,6))
    while (k<norbits):
        xtest=rng.uniform(-xmax,xmax)
        # now, see the available kinetic energy 
        T = Ej - pot_eff(potential,omega,np.array([xtest,0,0]))
        # create a velocity between this and the vmax
        vxmax = vx_zero_velocity_curve(xtest,potential,omega,Ej)
        vxtest = rng.uniform(-vxmax,vxmax)
        # give the remaining energy to the vy component
        additional_energy = T - (vxtest**2)/2
        vytest = np.sqrt(2*additional_energy)
        initial_conditions[k] = [xtest,0,0,vxtest,vytest,0]
        k+=1
    return initial_conditions



# --- initial conditions throughout the bar interior, bounded by the L1 jacobi energy ---

def lagrange_l1_energy(pot, omega, xmin=0.1, xmax=20):
    """Return (x_L1, E_L1): the L1 location and its effective potential."""
    myxL1 = xL1(pot, omega, xmin=xmin, xmax=xmax)
    return myxL1, float(pot_eff(pot, omega, np.array([myxL1, 0.0, 0.0])))

def ymax_within_bar(pot, omega, x, EL1, y_start):
    """Largest |y| in the z=0 plane, at fixed x, with pot_eff <= EL1 (connected to the origin)."""
    x = abs(x)
    # the potential ridge in y bounds the bracket, since pot_eff is not monotonic in y
    ridge = optimize.minimize(lambda y: -float(pot_eff(pot, omega, np.array([x, y[0], 0.0]))), y_start).x[0]
    return float(optimize.brentq(lambda y: pot_eff(pot, omega, np.array([x, y, 0.0])) - EL1, 0, ridge))

def zmax_within_bar(pot, omega, x, y, EL1, ymax):
    """Largest |z| at fixed (x, y) with pot_eff <= EL1; ymax is used as the upper bracket."""
    return float(optimize.brentq(lambda z: pot_eff(pot, omega, np.array([x, y, z])) - EL1, 0, ymax))

def available_speed(pot, omega, pos, EL1):
    """Speed at which the kinetic energy equals EL1 - pot_eff(pos)."""
    return float(np.sqrt(max(-2 * (pot_eff(pot, omega, np.asarray(pos, dtype=float)) - EL1), 0.0)))

def sample_velocity_isotropic(vmax, rng):
    """
    Uniform in the 3D velocity ball of radius vmax: p(v) ~ v^2, direction isotropic.
    Returns (velocity vector, v, theta, phi) with theta the polar angle in [0, pi].
    """
    v = vmax * rng.uniform() ** (1 / 3)
    theta = np.arccos(rng.uniform(-1, 1))
    phi = rng.uniform(-np.pi, np.pi)
    vec = v * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return vec, v, theta, phi

def _position_weight(pot, omega, pos, EL1, weighting, ymax, zmax):
    # correct the sequential density 1/(ymax*zmax) (and optionally the velocity-ball volume)
    w = ymax * zmax
    if weighting == "phase_space":
        w *= available_speed(pot, omega, pos, EL1) ** 3
    return w

def _weight_envelope(pot, omega, xl1, EL1, weighting, ngrid=64, safety=1.2):
    # estimated on the y=z=0 line; samples whose weight exceeds the envelope are clipped
    best = 0.0
    for x in np.linspace(-xl1, xl1, ngrid + 2)[1:-1]:
        ymax = ymax_within_bar(pot, omega, x, EL1, xl1 / 2)
        zmax = zmax_within_bar(pot, omega, x, 0.0, EL1, ymax)
        best = max(best, _position_weight(pot, omega, np.array([x, 0.0, 0.0]), EL1, weighting, ymax, zmax))
    return safety * best

def sample_position_within_bar(pot, omega, rng, xl1, EL1, weighting="none", envelope=None):
    """
    Sequential draw: x in [-xL1, xL1], then y in +-ymax(x), then z in +-zmax(x, y).
    weighting: "none" keeps that draw, "volume" makes positions uniform in volume,
    "phase_space" also weights by the available velocity-ball volume.
    """
    if weighting not in ("none", "volume", "phase_space"):
        raise ValueError(f"unknown weighting {weighting!r}")
    if weighting != "none" and envelope is None:
        envelope = _weight_envelope(pot, omega, xl1, EL1, weighting)
    while True:
        x = rng.uniform(-xl1, xl1)
        ymax = ymax_within_bar(pot, omega, x, EL1, xl1 / 2)
        y = rng.uniform(-ymax, ymax)
        zmax = zmax_within_bar(pot, omega, x, y, EL1, ymax)
        z = rng.uniform(-zmax, zmax)
        pos = np.array([x, y, z])
        if weighting == "none":
            return pos
        w = _position_weight(pot, omega, pos, EL1, weighting, ymax, zmax)
        if rng.uniform() < min(1.0, w / envelope):
            return pos

def to_inertial_velocity(ics, omega):
    """Bar-frame velocities -> the instantaneously coincident inertial frame that agama.orbit expects."""
    out = np.array(ics, dtype=float)
    out[..., 3] -= omega * ics[..., 1]
    out[..., 4] += omega * ics[..., 0]
    return out

def sample_phase_space_within_bar(pot, omega, norbits=50, seed=10, weighting="none", inertial=True):
    """Return (norbits, 6) array of [x, y, z, vx, vy, vz] bounded by the L1 jacobi energy.
    inertial=False returns the bar-frame velocities."""
    if weighting not in ("none", "volume", "phase_space"):
        raise ValueError(f"unknown weighting {weighting!r}")
    rng = np.random.default_rng(seed=seed)
    xl1, EL1 = lagrange_l1_energy(pot, omega)
    if weighting == "none":
        # slow sequential draw, kept for its non-uniform position density
        ics = np.zeros((norbits, 6))
        for k in range(norbits):
            pos = sample_position_within_bar(pot, omega, rng, xl1, EL1, weighting)
            vec, *_ = sample_velocity_isotropic(available_speed(pot, omega, pos, EL1), rng)
            ics[k] = np.concatenate([pos, vec])
        return to_inertial_velocity(ics, omega) if inertial else ics

    # batched rejection in the cube |x|,|y|,|z| < xL1, restricted to the sphere r < xL1 to stay connected to the origin
    positions, kinetic = [], []
    nkept, envelope, batch = 0, None, 100000
    while nkept < norbits:
        pos = rng.uniform(-xl1, xl1, size=(batch, 3))
        pos = pos[np.sum(pos**2, axis=1) < xl1**2]
        ke = EL1 - pot_eff(pot, omega, pos)
        pos, ke = pos[ke > 0], ke[ke > 0]
        if weighting == "phase_space":
            w = ke**1.5
            if envelope is None:
                envelope = 1.2 * w.max()  # later samples above this are clipped
            keep = rng.uniform(size=len(w)) < np.minimum(1.0, w / envelope)
            pos, ke = pos[keep], ke[keep]
        positions.append(pos)
        kinetic.append(ke)
        nkept += len(pos)
        # size the next batch from the observed acceptance
        batch = int(min(2e6, max(1e5, 2 * (norbits - nkept) * batch / max(len(pos), 1))))
    pos = np.concatenate(positions)[:norbits]
    vmax = np.sqrt(2 * np.concatenate(kinetic)[:norbits])
    # isotropic direction, speed uniform in the velocity ball (p(v) ~ v^2)
    speed = vmax * rng.uniform(size=norbits) ** (1 / 3)
    costheta = rng.uniform(-1, 1, size=norbits)
    sintheta = np.sqrt(1 - costheta**2)
    phi = rng.uniform(-np.pi, np.pi, size=norbits)
    vel = speed[:, None] * np.stack([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta], axis=1)
    ics = np.hstack([pos, vel])
    return to_inertial_velocity(ics, omega) if inertial else ics
