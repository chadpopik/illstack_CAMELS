import sys
sys.path.append('/home/jovyan/home')
from Basics import *
setplot(dark=True)

from illstack_CAMELS.CAMELS_example import profiles_expand_new


basepath = "/home/jovyan/PUBLIC_RELEASE/"
suite = 'IllustrisTNG'
# basepath = "/home/jovyan/"
# suite = 'TNG_L50'

subset = '1P'
sims = ['p1_n2']
snaps = [74]
# sims = [f"{sim}" for sim in np.arange(10)]
# snap = zdf.loc[np.abs(zdf.Redshift-0.55).idxmin(), '91 snapshots']


profs = ['gasdens', 'gaspth'] 

lims = [10e-3, 10]  # profile limits in Mpc
bins = 50  # number of bins
mhlims = [10**11.0, 10**15.0]  # Halo mass in Msun
mass_kind = 'halo'  # Use halo mass M200c or stellar mass for cuts


for sim in sims:
    for snap in snaps:
        haloprofs, haloprops, simparams = profiles_expand_new.stack(basepath, suite, subset, sim, snap, profs, lims, bins, mhlims, mass_kind)
        
        file_hdf5 = h5py.File(f"/home/jovyan/home/Profiles/{suite}/{subset}/{suite}_{subset}_{sim}_{snap}.hdf5", 'w')
    
        # Add all the sim parameters
        file_hdf5.create_group('Parameters')
        for p in simparams.keys():
            file_hdf5.create_dataset(f'Parameters/{p}', data=simparams[p])
    
        # Add all halo properties
        file_hdf5.create_group('Groups', track_order=True)
        for prop in haloprops.keys():
            file_hdf5.create_dataset(f'Groups/{prop}', data=haloprops[prop], track_order=True, dtype='float64')
    
        # Add all the profiles
        file_hdf5.create_group('Profiles', track_order=True)
        for prof in haloprofs.keys():
            file_hdf5.create_dataset(f"Profiles/{prof}",data=haloprofs[prof].astype(np.float64), track_order=True, dtype='float64')
        
        file_hdf5.close()
