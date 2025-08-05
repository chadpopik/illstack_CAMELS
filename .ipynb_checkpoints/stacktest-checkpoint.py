import sys
sys.path.append('/home/jovyan/home')
from Basics import *


from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial import cKDTree

sys.path.insert(0,'/home/jovyan/home/illstack_CAMELS/')
from illstack.CompHaloProperties import CompHaloProp


global_posp = None
global_weighted_vals = None

def process_halo(posh, rh, halo_idxs, binsedges, box, scaled_radius, search_radius, lims,Nb, Nprof, n_jobs):    
    rel = (global_posp[halo_idxs] - posh + box/2) % box - box/2  # position of the particles relative to the halo center with periodic BCs
    r = np.linalg.norm(rel, axis=1)  # radial distance from the particle to the halo
    if scaled_radius:  r = r / rh   # if scaled, divide by halo radius

    bin_idx = np.digitize(r, binsedges) - 1  # radial bin index for every particle
    valid = (bin_idx >= 0) & (bin_idx < Nb)
    bin_idx = bin_idx[valid]  # ensure particles are within the binedges
    halo_idxs = np.array(halo_idxs, dtype=int)  # Ensure integer type
    idx_valid = halo_idxs[valid]

    profile_i = np.zeros((Nb, Nprof))
    for j in range(Nprof):
        prof = np.bincount(bin_idx, weights=global_weighted_vals[j, idx_valid], minlength=Nb)
        profile_i[:, j] = prof

    count_prof = np.bincount(bin_idx, minlength=Nb)
    return profile_i, count_prof



def stacknew(
    posp: np.ndarray,
    vals: np.ndarray,
    weights: np.ndarray,
    volweight: np.ndarray,
    posh: np.ndarray,
    rh: np.ndarray,
    box, scaled_radius, search_radius, lims, bins, n_jobs
):


    # Get bin edges and calculate volumes
    CHP = CompHaloProp(lims, bins)
    binsedges, r_centers = CHP.radbins, CHP.BinCenter

    # Calculate dimenisons
    Nh, Nb, Nprof = posh.shape[0], len(binsedges) - 1, vals.shape[0]
    weighted_vals = vals * weights

    global global_posp
    global global_weighted_vals
    global_posp = posp
    global_weighted_vals = weighted_vals

    # Build a KDTree to only find particles within the halo search bounds
    print("Building KDTree...")
    start = time.time()
    tree = cKDTree(posp, boxsize=box)

    print(f"KDTree built in {time.time()-start:.2f}...")

    print("Stacking halos in parallel...")
    start = time.time()
    profiles, counts=[],[]
    for i in tqdm(range(Nh)):
        r=search_radius * rh[i] if scaled_radius else lims[1]
        halo_idx = tree.query_ball_point(posh[i], r, workers=n_jobs)
        profile_i, count_prof= process_halo(posh[i], rh[i], halo_idx, binsedges, box, scaled_radius, search_radius, lims,Nb, Nprof, n_jobs)
        profiles.append(profile_i)
        counts.append(count_prof)


    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(process_halo)(posh[i], rh[i], halo_idx, binsedges, box, scaled_radius, search_radius, lims,Nb, Nprof, n_jobs
    #     )
    #     for i in tqdm(range(Nh)))
    

    print(f"Halos stacked in {time.time()-start:.2f} s")

    profiles_tmp = np.zeros((Nh, Nb, Nprof))
    counts = np.zeros((Nh, Nb), dtype=int)

    for i, (prof_i, count_i) in enumerate(results):
        profiles_tmp[i] = prof_i
        counts[i] = count_i

    # Volume weighting
    shell_volumes = 4/3 * np.pi * (binsedges[1:]**3 - binsedges[:-1]**3)
    for j in range(Nprof):
        if volweight[j]:
            profiles_tmp[:, :, j] /= shell_volumes

    profiles = np.transpose(profiles_tmp, (2, 0, 1))  # (Nprof, Nh, Nb)

    return profiles, counts, r_centers
    