import numpy, matplotlib.pyplot as plt, agama, os
from astropy import constants as cnst 
from astropy import units as u 
from scipy.optimize import root,fsolve,brentq
import matplotlib.gridspec as gridspec
from matplotlib import patches
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

import linearized_jacobi_surface as ljs

def get_phi_and_jacobi_contour(L1,Hess_eff,pot_cluster,npoints=100):
    # get the jacobi surface 
    dx_jacobi, dy_jacobi = ljs.make_vector_field_that_is_aligned_with_the_eigenvectors(Hess_eff, L1, npoints)
    pos_jacobi = numpy.column_stack((dx_jacobi, dy_jacobi,0*dy_jacobi))
    phi = ljs.effective_potential_with_effective_tensor(pos_jacobi, Hess_eff, pot_cluster)
    dx_jacobi = numpy.reshape(dx_jacobi,(npoints,npoints))
    dy_jacobi = numpy.reshape(dy_jacobi, (npoints,npoints))
    phi = numpy.reshape(phi, (npoints,npoints))  
    return  dx_jacobi, dy_jacobi, phi 


def main():

     ############################################
    ############ PARAMETERS TO MODIFY ############
     ############################################
    
    massCluster =   1e5
    scaleRadius =   5e-3

    pos_cluster =   numpy.array([3,2,0])
    eccentricity = 0.5 # a number between 0 and 1, with 1 being circular
    norbits = 3
    
    nframes = int(1e3)

    ## SET UP THE PROBLEM 
    # set the units 
    unitL = u.kpc
    unitV = u.km/u.s
    unitM = u.Msun
    G=cnst.G.to(unitV**2 * (unitL/unitM)).value
    agama.setUnits(mass=1, length=1, velocity=1)

    # not modifyable parameters at this stage of development 
    outpath = "effective_potential_linearized/disc_anim/"
    fname = outpath+"/frame_{:04d}.png"
    os.makedirs(outpath, exist_ok=True)

    # MW potential 
    potential   =   "McMillan17.ini"
    path        =   agama.__path__[0] + "/data/" + "McMillan17.ini"
    pot         =   agama.Potential(path)

    pot_cluster =   agama.Potential(type="plummer", mass=massCluster, scaleRadius=scaleRadius)
    
    force   =   pot.eval(pos_cluster,acc=True)
    vc      =   numpy.sqrt(numpy.abs(numpy.dot(pos_cluster,force)))
    vel     =   eccentricity*vc*numpy.linalg.cross(pos_cluster,[0,0,1])/numpy.linalg.norm(pos_cluster)
    ic      =   numpy.hstack((pos_cluster,vel))
    Tcirc   =   pot.Tcirc(ic)
    timestamps,orb=agama.orbit(ic=ic,potential=pot,time=norbits*Tcirc, trajsize=int(nframes))


    # GET PHYSICAL VALUES THAT ARE REQUIRED FOR SETTING UP THE AXES 
    Omega = ljs.get_omega_from_orbit(orb[0])
    Hess_cen = ljs.hessian_centrifugal(Omega)
    Hess_mw = ljs.hessian_galactic_potential(pot,orb[0,0:3])
    Hess_eff = Hess_mw + Hess_cen
    L1,EL1 = ljs.get_L1(Hess_eff, pot_cluster)
    rJ_largest = numpy.linalg.norm(L1)

    # FIXED QUANTITIES  
    factor=2;nvec = 20 
    dx_vec          =       numpy.linspace(-factor*rJ_largest,factor*rJ_largest,nvec)
    DX_VEC,DY_VEC   =       numpy.meshgrid(dx_vec,dx_vec)
    pos_vector_field=       numpy.column_stack((DX_VEC.flatten(),DY_VEC.flatten(),0*DX_VEC.flatten()))

    ntheta = 100
    theta = numpy.linspace(0,2*numpy.pi,int(ntheta))
    cluter_circ=scaleRadius*numpy.column_stack((numpy.cos(theta), numpy.sin(theta)))


    ## INITIATE THE FIGURE
    fig=plt.figure(figsize=((8.25 -2),2/3 * (8.25 -2)))
    sim_index=0
    gs = gridspec.GridSpec(1,2,width_ratios = (1,2),wspace=1/50)
    axis=fig.add_subplot(gs[0])
    axis2=fig.add_subplot(gs[1])    

    # PLOT THINGS THAT WILL REMAIN CONSTANT 
    line,=axis.plot(orb[:,0],orb[:,1],color="k",linewidth=0.5)
    axis.set(aspect="equal",yticks=[],xticks=[])
    axis2.set(yticks=[],xticks=[],aspect='equal')
    fig.text(0.125,0.125,s="S. Ferrone 2026",fontdict={"size":8},ha="left",va="bottom")
    axis2.plot(cluter_circ[:,0],cluter_circ[:,1],color="k")
    axis.set(title="McMillan17 Milky Way")   

    # THE LOOP! 
    fname = "frame_{:04d}"
    SIM_INDEX = 100
    npoints = 100
    # update the tidal field 
    Hess_mw = ljs.hessian_galactic_potential(pot,orb[SIM_INDEX,0:3])
    Omega=ljs.get_omega_from_orbit(orb[SIM_INDEX])
    Hess_cen = ljs.hessian_centrifugal(Omega)
    Hess_eff = Hess_mw + Hess_cen

    L1,EL1 = ljs.get_L1(Hess_eff, pot_cluster)
    dx_jacobi, dy_jacobi, phi = get_phi_and_jacobi_contour(L1,Hess_eff,pot_cluster)

    f_norm  = ljs.get_normalized_vector_field(pos_vector_field, Hess_eff, pot_cluster)

    # update the plot 
    quiv    =   axis2.quiver(pos_vector_field[:,0],pos_vector_field[:,1],f_norm[:,0],f_norm[:,1])
    scat    =   axis.scatter(orb[SIM_INDEX,0],orb[SIM_INDEX,1],zorder=10,color="k")
    axis2.contour(dx_jacobi, dy_jacobi, phi, levels=[EL1[0]], colors="k",linestyle="-");
    fig.savefig(outpath+fname.format(SIM_INDEX), dpi=300)

if __name__=="__main__":
    main()



