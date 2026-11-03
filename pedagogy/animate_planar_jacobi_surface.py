""" Make the video (width/height forced to be even, which libx264 + yuv420p requires;
 the frames are 1875x1250 px, so the width is odd without the scale filter):
 ffmpeg -framerate 30 -i effective_potential_linearized/disc_anim/frame_%04d.png \
   -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -crf 18 \
   effective_potential_linearized/jacobi_surface.mp4
"""
import matplotlib
matplotlib.use("Agg")
import multiprocessing as mp
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


# per-process state, filled by init_worker
_W = {}

def get_phi_and_jacobi_contour(L1,Hess_eff,pot_cluster,npoints=100):
    # get the jacobi surface 
    dx_jacobi, dy_jacobi = ljs.make_vector_field_that_is_aligned_with_the_eigenvectors(Hess_eff, L1, npoints)
    pos_jacobi = numpy.column_stack((dx_jacobi, dy_jacobi,0*dy_jacobi))
    phi = ljs.effective_potential_with_effective_tensor(pos_jacobi, Hess_eff, pot_cluster)
    dx_jacobi = numpy.reshape(dx_jacobi,(npoints,npoints))
    dy_jacobi = numpy.reshape(dy_jacobi, (npoints,npoints))
    phi = numpy.reshape(phi, (npoints,npoints))  
    return  dx_jacobi, dy_jacobi, phi 

def init_worker(orb, pos_vector_field, massCluster, scaleRadius, outpath):
    """Runs once per process: build potentials and the figure that gets reused."""
    agama.setUnits(mass=1, length=1, velocity=1)
    pot = agama.Potential(agama.__path__[0] + "/data/McMillan17.ini")
    pot_cluster = agama.Potential(type="plummer", mass=massCluster, scaleRadius=scaleRadius)

    theta = numpy.linspace(0, 2*numpy.pi, 100)
    cluter_circ = scaleRadius*numpy.column_stack((numpy.cos(theta), numpy.sin(theta)))

    fig = plt.figure(figsize=((8.25 - 2), 2/3 * (8.25 - 2)))
    gs = gridspec.GridSpec(1, 2, width_ratios=(1, 2), wspace=1/50)
    axis = fig.add_subplot(gs[0])
    axis2 = fig.add_subplot(gs[1])

    # constant artists
    axis.plot(orb[:, 0], orb[:, 1], color="k", linewidth=0.5)
    axis.set(aspect="equal", yticks=[], xticks=[], title="McMillan17 Milky Way")
    axis2.set(yticks=[], xticks=[], aspect='equal', title="Jacobi Surface a Globular Cluster")
    fig.text(0.125, 0.125, s="S. Ferrone 2026", fontdict={"size": 8}, ha="left", va="bottom")
    axis2.plot(cluter_circ[:, 0], cluter_circ[:, 1], color="k")

    _W.update(dict(orb=orb, pos_vector_field=pos_vector_field, pot=pot,
                   pot_cluster=pot_cluster, fig=fig, axis=axis, axis2=axis2,
                   outpath=outpath, artists=[]))


def update_frame(sim_index, npoints=100):
    """Update only the time-dependent artists on this process's figure and save."""
    w = _W
    orb, pot, pot_cluster = w["orb"], w["pot"], w["pot_cluster"]
    axis, axis2, fig = w["axis"], w["axis2"], w["fig"]
    pos_vector_field = w["pos_vector_field"]

    # remove the previous frame's artists
    for art in w["artists"]:
        art.remove()
    w["artists"].clear()

    # update the tidal field
    Hess_mw = ljs.hessian_galactic_potential(pot, orb[sim_index, 0:3])
    Omega = ljs.get_omega_from_orbit(orb[sim_index])
    Hess_cen = ljs.hessian_centrifugal(Omega)
    Hess_eff = Hess_mw + Hess_cen

    L1, EL1 = ljs.get_L1(Hess_eff, pot_cluster)
    dx_jacobi, dy_jacobi, phi = get_phi_and_jacobi_contour(L1, Hess_eff, pot_cluster, npoints)
    f_norm = ljs.get_normalized_vector_field(pos_vector_field, Hess_eff, pot_cluster)

    # update the plot
    quiv = axis2.quiver(pos_vector_field[:, 0], pos_vector_field[:, 1], f_norm[:, 0], f_norm[:, 1])
    scat = axis.scatter(orb[sim_index, 0], orb[sim_index, 1], zorder=10, color="k")
    cont = axis2.contour(dx_jacobi, dy_jacobi, phi, levels=[EL1[0]], colors="k", linestyles="-")
    w["artists"].extend([quiv, scat, cont])   # ContourSet.remove() needs matplotlib >= 3.8

    fig.savefig(w["outpath"] + "frame_{:04d}.png".format(sim_index), dpi=300)
    return sim_index



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


    nproc = os.cpu_count()
    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc, initializer=init_worker,
                  initargs=(orb, pos_vector_field, massCluster, scaleRadius, outpath)) as pool:
        for i in pool.imap_unordered(update_frame, range(nframes), chunksize=4):
            pass  # optionally print progress


if __name__=="__main__":
    main()
