import schwarzschild, nbody

schwarzschild.main("ferrers_bar_10000.h5", 
                   "../../parameters/components/bar_ferrers_v1.ini",
                   "../../parameters/components/mw_fiducial_v1.ini",
                   norbits=2000, overwrite=True)

nbody.main("ferrers_bar_10000.h5", Nbody=10000, integrationtime_Gyr=2)
