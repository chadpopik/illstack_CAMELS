"""
This file handles fetching raw data from the binder: the simulations, halos, and parameters, both initial cosmology/astrophysics/sim box ones and parameters that are scanned over throughout the model.
"""


import sys
sys.path.append('/home/jovyan/home')
from Basics import *

import illustris_python.illustris_python as il



def check(suite, box, subset, sim, snap=None):  # Check if sim specified exists and return its base path
    # Special location for IllustrisTNG 50 boxes
    if box=='L50n512' and suite=='IllustrisTNG': path = "/home/jovyan/Sims"
    else: path = "/home/jovyan/PUBLIC_RELEASE/Sims"

    # Find available options in the directory
    suites = os.listdir(path)
    boxes = [box for box in os.listdir(f"{path}/{suite}/") if box in ['L25n256', 'L50n512']]
    subsets = [subset for subset in os.listdir(f"{path}/{suite}/{box}/") if subset not in ['L25n256', 'L50n512']]
    sims = [sim.replace(f'{subset}_', '') for sim in os.listdir(f"{path}/{suite}/{box}/{subset}/") if sim[-3:] not in ['txt', 'csv']]
    snaps = [int(snap.split('.')[0][-3:]) for snap in os.listdir(f"{path}/{suite}/{box}/{subset}/{subset}_{sim}") if snap[:4]=='snap']

    # Return an error if the option is not there
    if suite not in suites: 
        raise KeyError(f"Choose suite in {suites}")
    elif box not in boxes: 
        raise KeyError(f"Choose box in {boxes}")
    elif subset not in subsets:
        raise KeyError(f"Choose subset in {subsets}")
    elif sim not in sims:
        raise KeyError(f"Choose sim in {sims}")
    elif snap not in snaps and snap!=None:
        raise KeyError(f"Choose snap in {snaps}")

    return path


def getICs(suite, box, subset, sim, snap):  # Get ICs for the sim
    path = check(suite, box, subset, sim, snap)
    simpath = f"{path}/{suite}/{box}/{subset}/{subset}_{sim}"

    # Get parameters in attritbutes of files
    simICs = dict(h5py.File(il.snapshot.snapPath(simpath, snap), 'r')['Header'].attrs)
    haloICs = dict(h5py.File(il.groupcat.gcPath(simpath, snap), 'r')['Parameters'].attrs)

    # Fetch XH from the CAMB file
    for line in open(f"{simpath}/ICs/CAMB.params", 'r'):
        if line.strip().startswith("YHe"): 
            simICs['Xh'] = 1-np.float64(line.strip().split('=', 1)[-1])

    return simICs | haloICs


def getparams(suite, box, subset, sim, snap):  # Get parameters for the sim
    path = check(suite, box, subset, sim, snap)
    simpath = f"{path}/{suite}/{box}/{subset}/{subset}_{sim}"
    
    # Get astrophysical scanned parameters
    paramsdf = pd.read_csv(f"{'/'.join(simpath.split('/')[:-1])}/CosmoAstroSeed_{suite}_{box}_{subset}.txt", sep='\s+')  # Bit of a lazy fix
    return paramsdf.loc[paramsdf['#Name']==f"{subset}_{sim}"].to_dict('records')[0]


def getgasparticles(suite, box, subset, sim, snap, fields):  # Get Gas particles 
    path = check(suite, box, subset, sim, snap)
    return il.snapshot.loadSubset(f"{path}/{suite}/{box}/{subset}/{subset}_{sim}", snap, 'gas', fields=fields)


def gethalos(suite, box, subset, sim, snap):  # Get Halos
    path = check(suite, box, subset, sim, snap)
    return il.groupcat.loadHalos(f"{path}/{suite}/{box}/{subset}/{subset}_{sim}", snap)
