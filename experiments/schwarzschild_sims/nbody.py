import agama, numpy
from astropy import units as u 
from astropy import constants as cnst
import pathlib
import pyfalcon
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



def main():
    fname_phase = "ferrers_bar_nbody_100_phase_space.npy"
    fname_mass = "ferrers_bar_nbody_100_mass.npy"
    fname_out = "ferrers_bar_nbody_trajectories.npy"
    #### PARAMETERS 
    treeAngle = 0.1 # 0 means exact n body summation 
    # softening length as a fraction of the mean inter-particle-distance
    softening_frac = 1e-1
    # THE TIME STEP AND SAMPLING TIME WILL BE ADJUSTED TO BE EVENLY SPACED
    integrationtime = 1*u.Gyr
    timestep_frac   = 1/50
    samplingtime    = 0.1*u.Myr   
    # OUTFILE NAME 
    fname = ""


    # import the stuff
    path_bar = "/Users/ferrone/repos/bar-dynamics/parameters/components/bar_ferrers_v1.ini"
    path_ext = "/Users/ferrone/repos/bar-dynamics/parameters/components/mw_fiducial_v1.ini"
    parameters_ext = load_potential_parameters(path_ext)
    parameters_bar = load_potential_parameters(path_bar)
    barconfig = configparser.ConfigParser()
    barconfig.read(path_bar)

    # update the units
    oldUnitG    =   (u.km/u.s)**2 * (u.kpc/u.Msun)
    G           =   cnst.G.to(oldUnitG)
    unitM   =   parameters_bar["bar"]["mass"]*u.Msun
    unitL   =   parameters_bar["bar"]["scaleRadius"]*u.kpc
    unitV   =   (G*unitM/unitL)**(1/2)
    unitRot =   unitV/unitL
    unitT = unitL/unitV
    agama.setUnits(length=unitL.value, mass=unitM.value, velocity=unitV.value) 

    # rescale the parameters
    parameters_ext_rescaled = rescale_ini_dict(parameters_ext,unitM=unitM,unitL=unitL)
    parameters_bar_rescaled = rescale_ini_dict(parameters_bar,unitM=unitM,unitL=unitL)
    omega = numpy.float64(barconfig["orientation"]["pattern_speed_kms_kpc"]) * (u.km/u.s)/u.kpc
    omega = omega.to(unitRot).value 

    # create the agama potential 
    thin_disk   = agama.Potential(**parameters_ext_rescaled['thin_disk'])
    thick_disk  = agama.Potential(**parameters_ext_rescaled['thick_disk'])
    halo        = agama.Potential(**parameters_ext_rescaled['halo'])
    bar         = agama.Potential(**parameters_bar_rescaled["bar"])
    potential_external = thin_disk + thick_disk + halo

    # load the orbits 
    snap_mass = numpy.load(fname_mass)
    snap_posvel = numpy.load(fname_phase)
    nbody = snap_mass.shape[0]

    # compute the interparticle spacing for the softening length
    barvolume=numpy.pi*parameters_bar_rescaled["bar"]['scaleRadius']**3 * parameters_bar_rescaled["bar"]['axisRatioY'] * parameters_bar_rescaled["bar"]["axisRatioZ"]
    meandensity = bar.totalMass() / barvolume
    interparticledistance = (barvolume / nbody)**(1/3)
    eps = softening_frac * interparticledistance

    # ADJUST THE TIME STEPPING
    T  = integrationtime.to(unitT).value
    ts = samplingtime.to(unitT).value
    dt_max = timestep_frac*numpy.sqrt(eps**3 / snap_mass.mean())
    nsampling        = int(round(T/ts))                       # output has nsampling+1 rows
    steps_per_sample = int(numpy.ceil((T/nsampling)/dt_max))
    nsteps           = nsampling*steps_per_sample             # multiple of nsampling
    dt               = T/nsteps                               # <= dt_max
    samplingtime     = steps_per_sample*dt
    print("{:<25s} {:.2e}".format("integration time", (nsteps*dt*unitT).to(u.Myr)))
    print("{:<25s} {:.2e}".format("dt", (dt*unitT).to(u.Myr)))
    print("{:<25s} {:.2e}".format("sampling", (samplingtime*unitT).to(u.Myr)))
    print("{:<25s} {:.2e}".format("nsampling", nsampling))
    print("{:<25s} {:.2e}".format("nsteps", nsteps))

    # get the data volume 
    nphase = 6
    # get the data volume 
    dataVolume = (nbody*nphase*nsampling*64*u.bit).to(u.Gbyte)
    print("----+----+----+----+----+----+----+----+----+----")
    print("{:<25s} {:.2e}".format("Data volumne", dataVolume))

    # initialize! 
    potential_int   = numpy.zeros((nsampling+1,nbody))
    potential_ext   = numpy.zeros((nsampling+1,nbody))
    trajectory      = numpy.zeros((nsampling+1,nbody,nphase))
    vel = snap_posvel[:,3:6].copy()
    pos = snap_posvel[:,0:3].copy()
    i =0
    trajectory[0,:,:] = snap_posvel.copy()
    f_int,pot_int=pyfalcon.gravity(pos, snap_mass, eps=eps, theta=treeAngle)
    pot_ext, f_ext = potential_external.eval(pos,pot=True, acc=True)
    potential_int[i] = pot_int
    potential_ext[i] = pot_ext


    # DO THE INTEGRATION 
    for i in range(1,nsteps+1):
        # drift based on a half step 
        pos += 0.5 * dt * vel
        # now update the forces
        f_int,pot_int=pyfalcon.gravity(pos, snap_mass, eps=eps, theta=treeAngle)
        pot_ext, f_ext = potential_external.eval(pos,pot=True, acc=True)
        force = f_int+f_ext
        vel += dt * force
        # another half step 
        pos += 0.5 * dt * vel
        # I should update the potential to the new position, but check first
        if numpy.mod(i,steps_per_sample)==0:
            trajectory[int(i//steps_per_sample),:,0:3] = pos
            trajectory[int(i//steps_per_sample),:,3:6] = vel
            f_int,pot_int=pyfalcon.gravity(pos, snap_mass, eps=eps, theta=treeAngle)
            pot_ext, f_ext = potential_external.eval(pos,pot=True, acc=True)    
            potential_int[int(i//steps_per_sample)] = pot_int
            potential_ext[int(i//steps_per_sample)] = pot_ext 

    # SAVE THE OUTPUT IN A NUMPY ARRAY QUICK 
    numpy.save(fname_out,[trajectory,potential_int,potential_ext])





    
if __name__=="__main__":
     main()