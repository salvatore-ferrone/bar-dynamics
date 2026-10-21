import agama, numpy
from astropy import units as u 
from astropy import constants as cnst
import pathlib
import configparser


def load_potential_parameters(*ini_paths):
	parameters = {}
	for ini_path in ini_paths:
		config = configparser.ConfigParser()
		config.optionxform = str
		config.read(ini_path)
		for section in config.sections():
			if not section.startswith("Potential "):
				continue
			name = section.removeprefix("Potential ")
			parameters[name] = {
				key: value if key == "type" else float(value)
				for key, value in config[section].items()
			}
	return parameters

def rescale_ini_dict(parameters, unitM, unitL, unitM_OG=u.Msun, unitL_OG=u.kpc):
    """
    Expects the ini files to have been parsed with load_potential_parameters
    """
    parameters_rescaled = {}
    for key,value in parameters.items():
        parameters_rescaled[key] = {}
        for key_,value_ in value.items():
            if key_.startswith("mass"):
                parameters_rescaled[key][key_] = (value_*unitM_OG).to(unitM).value
            elif key_.startswith("scale"):
                parameters_rescaled[key][key_] = (value_*unitL_OG).to(unitL).value
            else:
                parameters_rescaled[key][key_] = value_    
    return parameters_rescaled


def loadconfigs(path_bar,path_ext):
    parameters_ext = load_potential_parameters(path_ext)
    parameters_bar = load_potential_parameters(path_bar)
    barconfig = configparser.ConfigParser()
    barconfig.read(path_bar)
    return parameters_ext, parameters_bar, barconfig

def main():

    #### PARAMETERS 

    Nbody = int(1e3)

    # tuning the orbital library 
    cross_times = 50
    beta = 1/2
    kappa = 1/10
    norbits = int(1e4)

    # tuning the density targets 
    nnodes          = 11
    rgridmax        = 1.2 
    rgridmin        = 0.1
    stripsPerPane   = 2

    fnameout = "ferrers_bar_nbody_{:d}".format(Nbody)
    virial_tolerence = 1/10
    path_bar = "/Users/ferrone/repos/bar-dynamics/parameters/components/bar_ferrers_v1.ini"
    path_ext = "/Users/ferrone/repos/bar-dynamics/parameters/components/mw_fiducial_v1.ini"

    parameters_ext, parameters_bar, barconfig = loadconfigs(path_bar,path_ext)

    # rescale the units 
    oldUnitG    =   (u.km/u.s)**2 * (u.kpc/u.Msun)
    G           =   cnst.G.to(oldUnitG)
    unitM   =   parameters_bar["bar"]["mass"]*u.Msun
    unitL   =   parameters_bar["bar"]["scaleRadius"]*u.kpc
    unitV   =   (G*unitM/unitL)**(1/2)
    unitRot =   unitV/unitL
    agama.setUnits(length=unitL.value, mass=unitM.value, velocity=unitV.value)     

    # rescale the parameters 
    parameters_ext_rescaled = rescale_ini_dict(parameters_ext,unitM=unitM,unitL=unitL)
    parameters_bar_rescaled = rescale_ini_dict(parameters_bar,unitM=unitM,unitL=unitL)
    omega = numpy.float64(barconfig["orientation"]["pattern_speed_kms_kpc"]) * (u.km/u.s)/u.kpc
    omega = omega.to(unitRot).value    

    # create the potential model in agama
    thin_disk   = agama.Potential(**parameters_ext['thin_disk'])
    thick_disk  = agama.Potential(**parameters_ext['thick_disk'])
    halo        = agama.Potential(**parameters_ext['halo'])
    bar         = agama.Potential(**parameters_bar["bar"])
    potential_external = thin_disk + thick_disk + halo
    potential_total = bar + potential_external 

    # sample the initial conditions 
    ics,mass    =   bar.sample(norbits, potential=potential_total, beta=beta,kappa=kappa)   

    # start getting the density targets 
    gridr           =   agama.nonuniformGrid(nnodes, rgridmin, rgridmax,)
    target_density  =   agama.Target(type = "DensityClassicLinear",
                                    gridr = gridr, 
                                    stripsPerPane = stripsPerPane, 
                                    axisRatioY=parameters_bar["bar"]['axisRatioY'], 
                                    axisRatioZ=parameters_bar["bar"]['axisRatioZ'])
    times           =   cross_times*potential_total.Tcirc(ics)    

    # create the orbital library 
    D,traj=agama.orbit(ic=ics,
                    potential=potential_total,
                    Omega=omega,
                    dtype=object,
                    time=times,
                    separateTime=True, 
                    targets=target_density) 


    # set up the schwarzchild model 
    total_bar_mass = bar.totalMass()
    w = numpy.ones(norbits) * total_bar_mass / norbits
    mass_discrete   = D.T.dot(w)
    mass_true       = target_density(bar) # the right hand side of the inversion     

    # solve the weights
    data0   = numpy.ones((1, norbits))
    sigma   = numpy.maximum(0.1*mass_true, 0.05*mass_true.mean())
    regul   = 1.0 
    weights = agama.solveOpt(
        matrix  = [data0, D.T],
        rhs     = [[total_bar_mass], mass_true],
        rpenq   = [[numpy.inf], 2*sigma**-2],
        xpenq   = regul * 2*numpy.ones(norbits)*norbits/total_bar_mass**2 
    )    

    # sample the initial conditions 
    snap_posvel, snap_mass = agama.sampleOrbitLibrary(Nbody, traj, weights)    

    # check the virial ratio 
    # approximate the potential with a multipole expansion 
    pot_stars_nbody =   agama.Potential(type='multipole', particles=(snap_posvel, snap_mass), symmetry='t')
    E_kin       =   0.5 * numpy.sum(snap_mass[:,None] * snap_posvel[:,3:6]**2)
    W_self      =   0.5 * numpy.sum(snap_mass * pot_stars_nbody.potential(snap_posvel[:,0:3]))
    force_rest  =   potential_external.force(snap_posvel[:,0:3])                      # -∇Φ_rest, shape (N,3)
    W_ext       =   numpy.sum(snap_mass * numpy.sum(snap_posvel[:,0:3]*force_rest, axis=1))
    virial_ratio = -2*E_kin / (W_self + W_ext)
    if numpy.abs(1-virial_ratio) > virial_tolerence: 
        print("Warning, virial ratio is above 10%. It is {:.3e}".format(virial_ratio))

    # Save the positions and velocities 
    numpy.save(fnameout+"_phase_space.npy", snap_posvel)
    numpy.save(fnameout+"_mass.npy", snap_mass)

    print("saved", fnameout)
    return None 

if __name__=="__main__":
     main()