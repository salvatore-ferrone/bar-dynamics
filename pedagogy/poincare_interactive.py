#!/usr/bin/python
'''
ADAPTED FROM AGAMA's EXAMPLE_POINCARE.PY
WILL DO FOR THE BAR-FRAME OF HUNTER ET AL (2024). 

'''

'''
This interactive example shows the meridional plane {x,z} (left panel) and the Poincare
surface of section {x,v_x} for an axisymmetric potential, where x is either cylindrical
radius R when L_z>0 or the x coordinate otherwise, and points on the SoS are placed
when passing through the z=0 plane with v_z>0.
Upon right-clicking at any point inside the zero-velocity curve on the surface of section,
a new orbit starting from these initial conditions is added to the plot.
The parameters of the potential, energy and L_z are specified at the beginning of the script.
'''
import agama, numpy, scipy.optimize, matplotlib, matplotlib.pyplot as plt
import numpy as np 
plt.rc('axes', linewidth=0.5)
plt.rc('font', size=8)
# consider motion in the x-z plane of an axisymmetric potential
# (with Lz=0 for motion in a flattened 2d potential, or Lz>0 for the motion in the meridional plane)
#pot  = agama.Potential(type='spheroid', gamma=1.5, q=0.5)


coleman_components=dict(\
    alpha = 0.626,
    c = 1.342 ,
    x1 = 0.49 ,
    y1 = 0.392,
    z1 = 0.229,
    xc = 0.751,
    yc = 0.469,
    cperp = 2.232,
    cparallel = 1.991,
    m = 0.873,
    n = 1.94,
    rcut = 4.37,
    rho_c = 3.16e9,)

portail_comp1 = dict(
    rho_i=0.5e9,    
    xi=5.364, 
    yi=0.959, 
    zi=0.611,
    R_in=0.558, 
    R_out=3.19,  
    cperp=0.97,  
    n_in=3.196, 
    n_out=16.731,  
    n_a=1.0)

portail_comp2=dict(
    rho_i=1.743e13, 
    xi=0.478, 
    yi=0.297, 
    zi=0.252,
    R_in=7.607, 
    R_out=2.204, 
    cperp=1.879, 
    n_in=1.63, 
    n_out=-27.291,
    n_a=1.0),

supermassiveblackhole = dict(type='Plummer', mass=4.154e6,scaleRadius=1e-4)
nsc = dict(type='Dehnen', mass=6.1e7, gamma=0.71, scaleRadius=5.9e-3, axisRatioZ=0.73)
NSD1 = dict(type='Spheroid', 
     densityNorm=1.311 * 153e10,
     gamma=0, 
     beta=0, 
     alpha=1, 
     scaleRadius=1,
     outerCutoffRadius=5.06e-3,
     cutoffStrength=0.72,
     axisRatioZ=0.37)

NSD2 = dict(type='Spheroid', 
     densityNorm=153e10,
     gamma=0, 
     beta=0, 
     alpha=1, 
     scaleRadius=1, # dummy, no power law comp
     outerCutoffRadius=24.6e-3,
     cutoffStrength=0.79,
     axisRatioZ=0.37)

DISK1 = dict(
    type = "Disk",
    surfaceDensity = 1.3719e3 * 1e6,
    scaleRadius = 2,
    scaleHeight = 300e-3,
    innerCutoffRadius=2.4
)
DISK2= dict(
    type = "Disk",
    surfaceDensity = 9.2391e2 * 1e6,
    scaleRadius = 2.8,
    scaleHeight = 900e-3,
    innerCutoffRadius=2.4
)

GDISK1 = dict(
    type = "Disk",
    surfaceDensity = 53.1 * 1e6,
    scaleRadius = 7,
    scaleHeight = -85e-3,
    innerCutoffRadius=4
)

GDISK2 = dict(
    type = "Disk",
    surfaceDensity = 2.18e3 * 1e6,
    scaleRadius = 1.2,
    scaleHeight = -45e-3,
    innerCutoffRadius=12
)

n=4.5
rs = 96
a = rs*(3*n-1/3)**(-n)
EINASTO = dict(type='Spheroid', mass=1.1e12,
               gamma=0, beta=0, alpha=1, scaleRadius=1,    # dummies: factor is 1
               outerCutoffRadius=a, cutoffStrength=1/n)     # axis ratios default to 1 (spherical)

def coleman2020(alpha, c, x1, y1, z1, xc, yc, cperp, cparallel, m, n, rcut, rho_c):
    # for agama, density normalization is one!
    # il numero di parametri in questo modello e' un offeso a dio ! opinione mio !
    def rho(xyz):
        x, y, z = xyz.T
        a = (((np.abs(x) / x1)**(cperp) + (np.abs(y) / y1)**(cperp))**(cparallel/cperp) + (np.abs(z)/z1)**cparallel)**(1/cparallel)
        aplus  = (((x+c*z)/xc)**2 + (y/yc)**2)**(1/2)
        aminus = (((x-c*z)/xc)**2 + (y/yc)**2)**(1/2)
        r = np.sqrt(x**2 + y**2 + z**2)
        return  (rho_c / np.cosh(a**m)) * (1 + alpha*(np.exp(-aplus**n) + np.exp(-aminus**n))) * np.exp(-(r/rcut)**2)
    return rho

# !!! THIS BAR IS A 20 PARAMETER MODEL!!!!
def portail_component(rho_i, xi, yi, zi, R_in, R_out, cperp, n_in, n_out, n_a):
    def rho(xyz):
        x, y, z = xyz.T
        R = np.sqrt(x**2 + y**2)
        a = ((np.abs(x)/xi)**cperp + (np.abs(y)/yi)**cperp)**(1/cperp)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            inner = np.exp(-(R_in/R)**n_in)
            outer = np.exp(-(R/R_out)**n_out)
        inner = np.nan_to_num(inner)
        outer = np.nan_to_num(outer)
        return rho_i * np.exp(-a**n_a) / np.cosh(z/zi)**2 * outer * inner
    return rho

def hunter_et_al_2024():
    bar1=coleman2020(**coleman_components)
    bar2=portail_component(**portail_components['component2'])
    bar3=portail_component(**portail_components['component3'])    
    pbar1   = agama.Potential(type='CylSpline', density=bar1,symmetry='triaxial',mmax=8,gridSizeR=40, gridSizeZ=40, Rmin=0.001, Rmax=20, Zmin=0.001, Zmax=20,)
    pbar2   = agama.Potential(type='CylSpline', density=bar2,symmetry='triaxial',mmax=8,gridSizeR=40, gridSizeZ=40, Rmin=0.001, Rmax=20, Zmin=0.001, Zmax=20,)
    pbar3   = agama.Potential(type='CylSpline', density=bar3,symmetry='triaxial',mmax=8,gridSizeR=40, gridSizeZ=40, Rmin=0.001, Rmax=20, Zmin=0.001, Zmax=20,)    
    psagA   = agama.Potential(**supermassiveblackhole)
    pnsc    = agama.Potential(**nsc)
    pnsd1   = agama.Potential(**NSD1)
    pnsd2   = agama.Potential(**NSD2)
    pdisk1  = agama.Potential(**DISK1)
    pdisk2  = agama.Potential(**DISK2)
    pgdisk1 = agama.Potential(**GDISK1)
    pgdisk2 = agama.Potential(**GDISK2)
    phalo   = agama.Potential(**EINASTO)
    return agama.Potential(pbar1,pbar2,pbar3,psagA,pnsc,pnsd1,pnsd2,pdisk1,pdisk2,pgdisk1,pgdisk2,phalo)

    

    
pot  = agama.Potential(type='disk', scaleheight=0.1)
rmax = 2.0
E    = pot.potential(rmax,0,0)
Lzmax= 2*numpy.pi * pot.Rcirc(E=E)**2 / pot.Tcirc(E)
Lz   = 0.1 * Lzmax

def init_axes(arg=None):
    axorb.cla()
    axpss.cla()
    axorb.set_xlim(0 if Lz>0 else -rmax, rmax)
    axorb.set_aspect('equal')
    axpss.set_xlim(axorb.get_xlim())
    axorb.set_xlabel('$x$', fontsize=12)
    axorb.set_ylabel('$z$', fontsize=12)
    axpss.set_xlabel('$x$', fontsize=12)
    axpss.set_ylabel('$p_x$', fontsize=12)
    # plot boundaries of orbit plane and surface of section
    Rp,Ra= pot.Rperiapo(E, Lz)
    xmin = -Ra if Lz==0 else Rp
    xmax = Ra
    grid = numpy.linspace(0, 1, 100)
    grid = grid * grid * (3-2*grid) * (xmax-xmin) + xmin
    vval = numpy.maximum(0, 2*E - 2*pot.potential(numpy.column_stack((grid,grid*0,grid*0))) - (Lz/grid)**2)**0.5
    zval = numpy.hstack([0,
        numpy.array([ scipy.optimize.brentq(lambda z: pot.potential(xx,0,z) - E + 0.5*(Lz/xx)**2, 0, xmax) for xx in grid[1:-1]]),
        0])
    axorb.plot(numpy.hstack((grid[:-1], grid[::-1])), numpy.hstack((zval[:-1], -zval[::-1])), color='k', lw=0.5)
    axpss.plot(numpy.hstack((grid[:-1], grid[::-1])), numpy.hstack((vval[:-1], -vval[::-1])), color='k', lw=0.5)
    axorb.text(0.5, 1.01, 'orbit plane',        ha='center', va='bottom', transform=axorb.transAxes, fontsize=10)
    axpss.text(0.5, 1.01, 'surface of section', ha='center', va='bottom', transform=axpss.transAxes, fontsize=10)
    plt.draw()

def run_orbit(ic):
    color = numpy.random.random(size=3)*0.8
    # create an orbit represented by a spline interpolator
    orbit = agama.orbit(ic=ic, potential=pot, time=100*pot.Tcirc(ic), dtype=object)
    # get all crossing points with z=0
    timecross = orbit.z.roots()
    # select those at which vz>=0
    timecross = timecross[orbit.z(timecross, der=1) >= 0]
    # get recorded trajectory sampled at every timestep...
    traj = orbit(orbit)
    # ...and at all crossing times
    trajcross = orbit(timecross)
    if Lz==0:
        axorb.plot(traj[:,0], traj[:,2], color=color, lw=0.5, alpha=0.5)
        axpss.plot(trajcross[:,0], trajcross[:,3], 'o', color=color, mew=0, ms=1.5)
    else:
        # orbit in the R,z plane, and SoS in the R, v_R plane
        axorb.plot((traj[:,0]**2 + traj[:,1]**2)**0.5, traj[:,2], color=color, lw=0.5, alpha=0.5)
        R = (trajcross[:,0]**2 + trajcross[:,1]**2)**0.5
        vR= (trajcross[:,0]*trajcross[:,3] + trajcross[:,1]*trajcross[:,4]) / R
        axpss.plot(R, vR, 'o', color=color, mew=0, ms=1.5)

def add_point(event):
    if event.inaxes is not axpss or event.button != 3: return
    x, vx = event.xdata, event.ydata
    vz2 = 2 * (E - pot.potential(x,0,0)) - (Lz/x)**2 - vx**2
    if vz2>0:
        run_orbit([x, 0, 0, vx, Lz/x, vz2**0.5])
        plt.draw()

fig   = plt.figure(figsize=(6,3), dpi=200)
axorb = plt.axes([0.08,0.14,0.4,0.8])
axpss = plt.axes([0.58,0.14,0.4,0.8])
button_clear = matplotlib.widgets.Button(plt.axes([0.90,0.88,0.08,0.06]), 'clear')
fig.canvas.mpl_connect('button_press_event', add_point)
button_clear.on_clicked(init_axes)
init_axes()
print('Right-click on the Surface of Section to start an orbit')
plt.show()
3