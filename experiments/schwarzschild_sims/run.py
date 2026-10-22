import schwarzschild, nbody

schwarzschild.main("ferrers_bar.h5", 
                   "../../parameters/components/bar_ferrers_v1.ini",
                   "../../parameters/components/mw_fiducial_v1.ini",
                   norbits=2000, overwrite=True)

nbody.main("ferrers_bar.h5", Nbody=200, integrationtime_Gyr=2)
