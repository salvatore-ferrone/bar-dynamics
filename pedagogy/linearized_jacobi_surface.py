import numpy, agama
from scipy.optimize import fsolve

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
    return -tidalTensor

def hessian_centrifugal(Omega):
    omega = numpy.linalg.norm(Omega)
    term1=omega**2 * numpy.eye(3)
    term2 = numpy.linalg.outer(Omega,Omega)
    return (term2 - term1)

def effective_potential_with_effective_tensor(pos, Hessian, pot_cluster):  # p: (N,3)
    pos = numpy.atleast_2d(pos)
    quad = 0.5*numpy.einsum('ni,ij,nj->n', pos, Hessian, pos)
    return pot_cluster.potential(pos) + quad

def F_total_cluster_frame(pos, Hessian, pot_cluster):
    F_hess=-(Hessian@pos.T).T
    F_gc = pot_cluster.eval(pos,acc=True)
    return F_hess+F_gc

def get_omega_from_orbit(posvel):
    return numpy.linalg.cross(posvel[0:3],posvel[3:6]) / numpy.linalg.norm(posvel[0:3])**2

def get_L1(Hess_eff,pot_cluster):
    eigenvalues, eigenvectors   =   numpy.linalg.eigh(Hess_eff)
    eigenvector                 =   eigenvectors[:, 0]
    
    eigenvector/=numpy.linalg.norm(eigenvector)
    lambda1 = -eigenvalues[0]
    
    rJ = (agama.G*pot_cluster.totalMass()/lambda1)**(1/3)
    rJ_guess = rJ * eigenvector # makes it a vector
    
    L1 = fsolve(F_total_cluster_frame,rJ_guess,args=(Hess_eff,pot_cluster))
    EL1 =effective_potential_with_effective_tensor(L1,Hess_eff,pot_cluster)
    return L1, EL1

def make_vector_field_that_is_aligned_with_the_eigenvectors(Hess_eff,L1,npoints=100):
    "Because, I don't want the surface to extend beyond the jacobi surface"
    _, eigenvectors   =   numpy.linalg.eigh(Hess_eff)
    rJ = numpy.linalg.norm(L1)
    # tilt the football so we only get the axis and don't go beyond for the jacobi surface
    dx_phi = numpy.linspace(-rJ,rJ,npoints)
    dy_phi = numpy.linspace(-rJ,rJ,npoints)
    DX,DY = numpy.meshgrid(dx_phi,dy_phi)
    pos_phi = numpy.column_stack((DX.flatten(), DY.flatten(), 0*DY.flatten()))
    # alle the rotation matric of the eivenvectors 
    pos_phi = (eigenvectors @ pos_phi.T).T
    DX = pos_phi[:,0]; DY=pos_phi[:,1]
    return DX,DY

def get_normalized_vector_field(pos, Hess, pot_cluster):
    F_eff =   -(Hess@pos.T).T
    F_gc  =   pot_cluster.eval(pos,acc=True)
    F_tot =   F_eff+F_gc
    F_tot_norm = numpy.divide(F_tot.T,numpy.linalg.norm(F_tot,axis=1),).T
    return F_tot_norm