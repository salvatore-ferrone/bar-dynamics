'''
This file initializes the agama potential for the Hunter et al 2024 Milky Way potential model 

'''
import numpy as np 
import agama
agama.setUnits(length=1,mass=1,velocity=1)

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

portail_components = {
    "component2": dict(rho_i=0.5e9,    
                       xi=5.364, 
                       yi=0.959, 
                       zi=0.611,
                       R_in=0.558, 
                       R_out=3.19,  
                       cperp=0.97,  
                       n_in=3.196, 
                       n_out=16.731,  
                       n_a=1.0),
    "component3": dict(rho_i=1.743e13, 
                       xi=0.478, 
                       yi=0.297, 
                       zi=0.252,
                       R_in=7.607, 
                       R_out=2.204, 
                       cperp=1.879, 
                       n_in=1.63, 
                       n_out=-27.291,
                       n_a=1.0),}

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

def mw_model():
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
