import healpy as hp
import numpy as np
cimport numpy as np
import time

from illstack.CompHaloProperties import CompHaloProp

# Set the edges of the box to connect to the other side
def periodic_bcs(np.ndarray posp,np.ndarray posh, box):
    # Position of particles
    xp, yp, zp = posp[:,0], posp[:,1], posp[:,2]

    # Position of the halos
    xh, yh, zh= posh[0], posh[1], posh[2]

    # Distance between halo and particles
    xdel, ydel, zdel = xp - xh, yp - yh, zp - zh

    # If the particle is more than half the box size away, move it to the other side of the halo
    xp[xdel >= box/2.] = xp[xdel >= box/2.] - box
    yp[ydel >= box/2.] = yp[ydel >= box/2.] - box
    zp[zdel >= box/2.] = zp[zdel >= box/2.] - box
    xp[xdel < -1.*box/2.] = xp[xdel < -1. *box/2.] + box 
    yp[ydel < -1.*box/2.] = yp[ydel < -1. *box/2.] + box 
    zp[zdel < -1.*box/2.] = zp[zdel < -1. *box/2.] + box 
    
    posp=np.column_stack([xp,yp,zp])

    del xp, yp, zp, xh, yh, zh, xdel, ydel, zdel

    return posp


def add_ghost_particles(posc,vals_init,weights_init,maxrad, box): #this does periodic bcs
    # Add the original point values
    posc_ghosts, vals_ghosts, weights_ghosts = [posc], [vals_init], [weights_init]

    # Set limits for how far outside the box the search will extend
    x1, y1, z1 = -maxrad, -maxrad, -maxrad
    x2, y2, z2 = box + maxrad, box + maxrad, box + maxrad

    # For every direction it can be extended
    for i in (-1,0,1):
        for j in (-1,0,1):
            for k in (-1,0,1):
                print(i, j, k)
                if (i==0 and j==0 and k==0):  # Avoids doubling the original points
                    continue
                xp, yp, zp = posc[:,0] + i*box, posc[:,1] + j*box, posc[:,2] + k*box
                dm = (xp>x1) & (xp<x2) & (yp>y1) & (yp<y2) & (zp>z1) & (zp<z2)

                # Append the particle values
                if np.any(dm):  # Only process if there are valid points
                    posc_ghosts.append(np.column_stack([xp[dm], yp[dm], zp[dm]]))
                    vals_ghosts.append(vals_init[:, dm])
                    weights_ghosts.append(weights_init[:, dm])

    # Properly stack them
    posc_ghosts = np.vstack(posc_ghosts)
    vals_ghosts = np.hstack(vals_ghosts)
    weights_ghosts = np.hstack(weights_ghosts)
    return posc_ghosts, vals_ghosts, weights_ghosts


def cull_and_center(np.ndarray posp, np.ndarray vals, np.ndarray weights, 
                    np.ndarray posh, rhi, rbuff, scaled_radius, search_radius, box):

    # Set the position of the particle with the halo as the center
    xp = posp[:,0]-posh[0]; yp=posp[:,1]-posh[1]; zp=posp[:,2]-posh[2]
    # Calculate the radial distance of the particle from the halo center
    r = np.sqrt(xp**2+yp**2+zp**2)

    # Find particles only within the search radius
    if (scaled_radius == True): 
        dm = [r/rhi < search_radius]
    else:
        dm = [r < rbuff]
    dm = np.array(dm[0])
    posp = np.column_stack([xp[dm], yp[dm], zp[dm]])
    return posp, vals[:,dm], weights[:,dm]


def precull(np.ndarray posp, np.ndarray vals, np.ndarray weights, 
            np.ndarray posh, np.ndarray rh, search_radius):

    nchain = 256
    rbuff = rh.max() * search_radius

    x1 = posp[:,0].min()-1.1*rbuff; x2 = posp[:,0].max()+1.1*rbuff
    y1 = posp[:,1].min()-1.1*rbuff; y2 = posp[:,1].max()+1.1*rbuff
    z1 = posp[:,2].min()-1.1*rbuff; z2 = posp[:,2].max()+1.1*rbuff

    dx = (x2-x1) / nchain; dy = (y2-y1) / nchain; dz = (z2-z1) / nchain

    mask = np.reshape(np.zeros(nchain**3),(nchain,nchain,nchain)).astype(np.bool)
    for ih in np.arange(len(rh)):
        #print ih,len(rh)
        xl = posh[ih,0] - rbuff; xh = posh[ih,0] + rbuff
        yl = posh[ih,1] - rbuff; yh = posh[ih,1] + rbuff
        zl = posh[ih,2] - rbuff; zh = posh[ih,2] + rbuff
        il = max(int((xl-x1)/dx),0); ih = min(int((xh-x1)/dx),nchain-1)
        jl = max(int((yl-y1)/dy),0); jh = min(int((yh-y1)/dy),nchain-1)
        kl = max(int((zl-z1)/dz),0); kh = min(int((zh-z1)/dz),nchain-1)
        for i in np.arange(il,ih+1):
            for j in np.arange(jl,jh+1):
                for k in np.arange(kl,kh+1):
                    mask[i,j,k] = True

    pmask = np.zeros(len(vals)).astype(np.bool)
    for ip in np.arange(len(vals)):
        if ip%10000==0: print ip,len(vals)
        x = posp[ip,0]; y = posp[ip,1]; z = posp[ip,2]
        i = int((x-x1)/dx)
        j = int((y-y1)/dy)
        k = int((z-z1)/dz)
        if mask[i,j,k]: pmask[ip] = mask[i,j,k]

    posp    = posp[pmask]
    vals    = vals[pmask]
    weights = weights[pmask]

    return posp,vals,weights



def stackonhalostile(
        np.ndarray          pospi,  # Position of particles, [n_particles, 3]
        np.ndarray          valsi,  # Values of particles, [n_profs, n_particles]
        np.ndarray          poshi,  # Position of haloes [n_halos]
        np.ndarray      volweight,  # Volume weight truth of profiles, [n_profs]
        np.ndarray        weightsi,  # Weight on particles, [n_profs, n_particles]
        np.ndarray            rhi,  # Radii of haloes [n_halos]
        it, jt, kt, ntile, scaled_radius, search_radius, lims, bins, box, rank):

    CHP = CompHaloProp(lims,bins)
    # If using scaled radius, find extent of largest halo as extension bounds, else just use upper lims
    rbuff = search_radius * rhi.max() if scaled_radius else lims[1]
    
    x1=0.; x2=box; y1=0.; y2=box; z1=0.; z2=box;  # Bounds of the box
    dx, dy, dz = (x2-x1)/ntile, (y2-y1)/ntile, (z2-z1)/ntile;  # Width of each tile
    
    # Set bounds for halo so they exist within the tile
    xh, yh, zh = poshi[:,0], poshi[:,1], poshi[:,2]  # Position of halos
    x1h, y1h, z1h = it*dx, jt*dy, kt*dz  # Lower bound for halos
    x2h, y2h, z2h = (it+1)*dx, (jt+1)*dy, (kt+1)*dz  # Upper bound for halos
    dmh = (xh>x1h) & (xh<x2h) & (yh>y1h) & (yh<y2h) & (zh>z1h) & (zh<z2h)
    posh  = np.column_stack([xh[dmh], yh[dmh], zh[dmh]])
    rh = rhi[dmh] 

    nhalos=np.sum(dmh)
    if rank==0: #when MPI isn't initialized
        print it*ntile**2+jt*ntile+kt+1,'of',ntile**3,'done, nhalos =',nhalos

    # If there are no halos, return the empty arrays
    if nhalos == 0:
        return np.empty((len(volweight),0),float), np.empty((len(volweight),0),float), np.empty((len(volweight),0),float)
    
    # Set bounds of particles so they exist within one max search radius of the tile
    xp, yp, zp= pospi[:,0], pospi[:,1], pospi[:,2]  # Position of particles
    x1p, y1p, z1p = x1h-rbuff, y1h-rbuff, z1h-rbuff  # Lower bound of particles
    x2p, y2p, z2p = x2h+rbuff, y2h+rbuff, z2h+rbuff  # Upper bound for particles
    
    dmp = (xp>x1p) & (xp<x2p) & (yp>y1p) & (yp<y2p) & (zp>z1p) & (zp<z2p)
    posp  = np.column_stack([xp[dmp], yp[dmp], zp[dmp]])
    vals, weights = valsi[:,dmp], weightsi[:,dmp]

    pcen_list, pval_list, pnum_list = [], [], []

    for ih in range(nhalos):
            print ih, "out of", nhalos
            pospc, valsc, weightsc = cull_and_center(posp,vals,weights,posh[ih],rhi[ih],rbuff,scaled_radius=scaled_radius, search_radius=search_radius, box=box)
            scale=rh[ih]
            pcenc, pvalc, pnumc = CHP.ComputeHaloProfile(pospc,valsc,weightsc,scale,volweight=volweight,scaled_radius=scaled_radius)
            # pcen = np.append(pcen,pcenc)
            # pval = np.append(pval,pvalc,axis=1) #here
            # pnum = np.append(pnum,pnumc,axis=1) #here
            pcen_list.append(pcenc)
            pval_list.append(pvalc)
            pnum_list.append(pnumc)
            
    pcen, pval, pnum = np.hstack(pcen_list), np.hstack(pval_list), np.hstack(pnum_list)
    
    return pcen,pval,pnum


def stackonhalos(
        np.ndarray          posp,  # Position of particles, [n_particles, 3]
        np.ndarray          vals,  # Values of particles, [n_profs, n_particles]
        np.ndarray       weights,  # Weight on particles, [n_profs, n_particles]
        np.ndarray     volweight,  # Volume weight truth of profiles, [n_profs]
        np.ndarray          posh,  # Position of haloes [n_halos]
        np.ndarray            rh,  # Radii of haloes [n_halos]
        box, scaled_radius, search_radius, lims, bins):

    # ntile = np.floor((rh.size/5)**(1/3))  # controls tiling -- optimal when each tile has a few halos
    ntile=3
    rank = 0  # print messages about halos 

    print("Adding ghost particles for periodic BCs")
    # If using scaled radius, find extent of largest halo as extension bounds, else just use upper lims
    rbuff = search_radius * rh.max() if scaled_radius else lims[1]
    posp,vals,weights = add_ghost_particles(posp,vals,weights,rbuff, box)

    print("Stacking Halos")
    pcen_list, pval_list, pnum_list = [], [], []    
    for it in np.arange(ntile):
        for jt in np.arange(ntile):
            for kt in np.arange(ntile):
                pcenc, pvalc, pnumc= stackonhalostile(posp,vals,posh,volweight,weights,rh,it,jt,kt,ntile,scaled_radius, search_radius, lims, bins, box, rank)   
                pcen_list.append(pcenc)
                pval_list.append(pvalc)
                pnum_list.append(pnumc)

    pcen = np.concatenate(pcen_list) if pcen_list else np.empty((0,))
    pval = np.concatenate(pval_list, axis=1) if pval_list else np.empty((vals.shape[0], 0))
    pnum = np.concatenate(pnum_list, axis=1) if pnum_list else np.empty((vals.shape[0], 0))
            
    return pval, pnum, pcen