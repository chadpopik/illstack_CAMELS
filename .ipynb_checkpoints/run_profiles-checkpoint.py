"""
This file is reponsible for actaully running the profile stacking process
"""

import sys
sys.path.append('/home/jovyan/home')
from Basics import *

from open_sims import getparams
from make_profiles import stack

suite = 'IllustrisTNG'  # Which Suite
box = 'L25n256'
subset = 'SB28'
sims = ['2']
snaps = [24]
# sims = [f"{sim}" for sim in np.arange(10)]
# snap = zdf.loc[np.abs(zdf.Redshift-0.55).idxmin(), '91 snapshots']

profs = ['gasdens', 'gaspth']

mass_kind = 'halo'  # Use halo mass M200c or stellar mass for cuts
mhlims = [10**11, 10**15.0]  # Halo mass in Msun

scaled_radius = None  # Whether to use scaled radius x=r/r200c
rlims = [1e-2, 1e1]  # profile limits in Mpc
rbins = 50  # number of bins

n_jobs = 1

savepath = "/home/jovyan/home/illstack_CAMELS/Profiles"



for sim in sims:
    path = f"{savepath}/{suite}/{box}/{subset}/{sim}"
    if os.path.isdir(path) is False:
        os.makedirs(path)
        
    params = getparams(suite, box, subset, sim, snap=snaps[0])
    for snap in snaps:
        haloprofs, haloprops, ICs = stack(suite, box, subset, sim, snap, profs, mhlims, mass_kind, rlims, scaled_radius, rbins, n_jobs)

        file_hdf5 = h5py.File(f"{path}/{suite}_{box}_{subset}_{sim}_{snap}.hdf5", 'w')
        
        # Add all the profiles
        file_hdf5.create_group('Profiles', track_order=True)
        for prof in haloprofs.keys():
            file_hdf5.create_dataset(f"Profiles/{prof}",data=haloprofs[prof].astype(np.float64), track_order=True, dtype='float64')

        # Add all halo properties
        file_hdf5.create_group('Groups', track_order=True)
        for prop in haloprops.keys():
            file_hdf5.create_dataset(f'Groups/{prop}', data=haloprops[prop], track_order=True, dtype='float64')

        # Add all the sim ICs
        file_hdf5.create_group('ICs')
        for IC in ICs.keys():
            file_hdf5.create_dataset(f'ICs/{IC}', data=ICs[IC])

        # Add all the parameters
        file_hdf5.create_group('Parameters')
        for param in params.keys():
            file_hdf5.create_dataset(f'Parameters/{param}', data=params[param])

        # Add profile setup as attributes
        file_hdf5.attrs["mass_kind"] = mass_kind
        file_hdf5.attrs["mhlims"] = mhlims
        file_hdf5.attrs["scaled_radius"] = "None" if scaled_radius is None else scaled_radius
        file_hdf5.attrs["rlims"] = rlims
        file_hdf5.attrs["rbins"] = rbins
        
        file_hdf5.close()
