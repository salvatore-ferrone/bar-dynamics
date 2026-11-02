import numpy, matplotlib.pyplot as plt, agama, os
from astropy import constants as cnst 
from astropy import units as u 
from scipy.optimize import root,fsolve,brentq

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


# construct the tidal tensor 
def hessian_galactic_potential(pot, pos):
    dFxdx,dFydy,dFzdz,dFxdy,dFydz,dFzdx=pot.eval(pos,der=True).T
    dFydx       =   dFxdy
    dFxdz       =   dFzdx
    dFzdy       =   dFydz
    column1     =   numpy.array([dFxdx, dFydx, dFzdx])
    column2     =   numpy.array([dFxdy, dFydy, dFzdy])
    column3     =   numpy.array([dFxdz, dFydz, dFzdz])
    tidalTensor =   numpy.column_stack([column1,column2,column3])
    return -(1/2)*tidalTensor

def hessian_centrifugal(Omega):
    omega = numpy.linalg.norm(Omega)
    term1=omega**2 * numpy.eye(3)
    term2 = numpy.linalg.outer(Omega,Omega)
    return (1/2)*(term2 - term1)

def Phi_eff(pos, Hess_cen, Hess_mw, pot_cluster):  # p: (N,3)
    pos = numpy.atleast_2d(pos)
    quad = 0.5*numpy.einsum('ni,ij,nj->n', pos, Hess_cen + Hess_mw, pos)
    return pot_cluster.potential(pos) + quad

def F_total_cluster_frame(pos, pot, omega, poscluster, pot_cluster):
    Hess_cen=hessian_centrifugal(omega)
    Hess_mw =hessian_galactic_potential(pot, poscluster)
    F_cen=-(Hess_cen@pos.T).T
    F_tidal=-(Hess_mw@pos.T).T
    F_gc = pot_cluster.eval(pos,acc=True)
    return F_cen+F_tidal+F_gc

def get_omega_from_orbit(posvel):
    return numpy.linalg.cross(posvel[0:3],posvel[3:6]) / numpy.linalg.norm(posvel[0:3])**2




