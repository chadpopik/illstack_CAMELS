# cython: boundscheck=False, wraparound=False, nonecheck=False
from cython.parallel import prange
cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt, fmod
import omp
from libc.stdio cimport printf



cdef int n_threads = 4  # Number of threads (cores)

cdef extern from "omp.h":
    void omp_set_num_threads(int num_threads)


cdef inline double periodic_distance(double x1, double x2, double box) nogil:
    cdef double dx = x1 - x2
    dx = fmod(dx + box/2.0, box) - box/2.0
    return dx

@cython.boundscheck(False)
@cython.wraparound(False)
def stack_profiles(
    double[:, :] posp,           # (Np,3)
    double[:, :, :] vals,        # (Nfeat, Nh, Np)
    double[:, :, :] weights,     # (Nfeat, Nh, Np)
    int[:] volweight,            # (Nfeat,)
    double[:, :] posh,           # (Nh,3)
    double[:] rh,                # (Nh,)
    double[:] bins,              # (Nb+1,)
    double box,
    bint scaled_radius,
    double search_radius
):
    cdef int Nh = posh.shape[0]
    cdef int Nb = bins.shape[0] - 1
    cdef int Nfeat = vals.shape[0]
    cdef int Np = posp.shape[0]

    cdef double[:, :, :] profiles = np.zeros((Nfeat, Nh, Nb), dtype=np.float64)
    cdef int[:, :] counts = np.zeros((Nh, Nb), dtype=np.int32)

    cdef int i, j, k, b
    cdef double dx, dy, dz, r, r_scaled
    cdef double cx, cy, cz, rhalo, Rmax

    omp_set_num_threads(n_threads)  # Set number of threads globally
    # Parallel loop over halos
    for i in prange(Nh, nogil=True):
        cx = posh[i, 0]
        cy = posh[i, 1]
        cz = posh[i, 2]
        rhalo = rh[i]
        if scaled_radius:
            Rmax = search_radius * rhalo
        else:
            Rmax = search_radius

        for k in range(Np):
            dx = periodic_distance(posp[k, 0], cx, box)
            dy = periodic_distance(posp[k, 1], cy, box)
            dz = periodic_distance(posp[k, 2], cz, box)
            r = sqrt(dx*dx + dy*dy + dz*dz)
            if scaled_radius:
                r_scaled = r / rhalo
            else:
                r_scaled = r

            if r_scaled < Rmax:
                for b in range(Nb):
                    if bins[b] <= r_scaled < bins[b+1]:
                        counts[i, b] += 1
                        for j in range(Nfeat):
                            profiles[j, i, b] += vals[j, i, k] * weights[j, i, k]
                        break

    # Volume weight profiles if requested
    shell_volumes = 4.0/3.0 * 3.141592653589793 * (np.asarray(bins)[1:]**3 - np.asarray(bins)[:-1]**3)
    for j in range(Nfeat):
        if volweight[j]:
            for i in range(Nh):
                for b in range(Nb):
                    profiles[j, i, b] /= shell_volumes[b]


    return profiles, counts
    # return profiles, 0.5*(bins[1:] + bins[:-1]), counts
