'''
helper functions for the bar frame 
'''

import numpy as np 
import agama 
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

