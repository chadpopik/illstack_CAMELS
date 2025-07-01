import healpy as hp
import numpy as np
cimport numpy as np
# import params
# from CAMELS_example import istk_params as params


from illstack.CompHaloProperties import CompHaloProp
#search_radius = params.search_radius
# box = 25000. # NEED TO FIX THIS!!!!

def periodic_bcs(np.ndarray posp,np.ndarray posh, search_radius, box):
    
    xp = posp[:,0]
    yp = posp[:,1]
    zp = posp[:,2]

    xh = posh[0]
    yh = posh[1]
    zh = posh[2]

    xdel = xp - xh
    ydel = yp - yh
    zdel = zp - zh

    xp[xdel >= box/2.] = xp[xdel >= box/2.] - box
    xp[xdel < -1.*box/2.] = xp[xdel < -1. *box/2.] + box 
    yp[ydel >= box/2.] = yp[ydel >= box/2.] - box
    yp[ydel < -1.*box/2.] = yp[ydel < -1. *box/2.] + box 
    zp[zdel >= box/2.] = zp[zdel >= box/2.] - box
    zp[zdel < -1.*box/2.] = zp[zdel < -1. *box/2.] + box 
    
    posp=np.column_stack([xp,yp,zp])

    return posp

def add_ghost_particles(posc,vals_init,weights_init,maxrad, box): #this does periodic bcs
    posc_ghosts = [posc]
    vals_ghosts = [vals_init]
    weights_ghosts = [weights_init]

    x1, y1, z1 = -maxrad, -maxrad, -maxrad
    x2, y2, z2 = box + maxrad, box + maxrad, box + maxrad

    for i in (-1,0,1):
        for j in (-1,0,1):
            for k in (-1,0,1):
                if (i==0 and j==0 and k==0): 
                    continue
                xp = posc[:,0] + i*box
                yp = posc[:,1] + j*box
                zp = posc[:,2] + k*box
                dm = (xp>x1) & (xp<x2) & (yp>y1) & (yp<y2) & (zp>z1) & (zp<z2)
                print(i,j,k)

                if np.any(dm):  # Only process if there are valid points
                    posc_ghosts.append(np.column_stack([xp[dm], yp[dm], zp[dm]]))
                    vals_ghosts.append(vals_init[:, dm])
                    weights_ghosts.append(weights_init[:, dm])

                del xp, yp, zp, dm

    posc_ghosts = np.vstack(posc_ghosts)
    vals_ghosts = np.hstack(vals_ghosts)
    weights_ghosts = np.hstack(weights_ghosts)
    #print("vals_ghosts after", np.shape(vals_ghosts))
    print("Finished adding ghost particles for periodic boundary conditions")
    return posc_ghosts, vals_ghosts, weights_ghosts


def cull_and_center(np.ndarray posp, np.ndarray vals, np.ndarray weights, 
                    np.ndarray posh, rh,scaled_radius, search_radius, box):

    xp = posp[:,0]-posh[0]; yp=posp[:,1]-posh[1]; zp=posp[:,2]-posh[2]
    r = np.sqrt(xp**2+yp**2+zp**2)
    if (scaled_radius == True): 
        dm = [r/rh < search_radius]
    else:
        dm = [r < search_radius * rh]
    dm = np.array(dm[0])
    posp = np.column_stack([xp[dm], yp[dm], zp[dm]])
    return posp,vals = vals[:,dm], weights[:,dm]

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
        np.ndarray          pospi,
        np.ndarray          valsi,
        np.ndarray          poshi,
        np.ndarray      volweight, #here
        np.ndarray        weightsi, #here
        np.ndarray            mhi,
        np.ndarray            rhi,
        np.ndarray         mstari,
        dict halosprops,
        it, jt, kt,ntile,mhmin, mhmax,scaled_radius,mass_kind, lims, bins, search_radius, box, rank):

    '''
    Parameters
	particles[nparticles][npartprops]
        halos[nphalos][nhaloprops]

    Returns
	profiles[:,nhalos]
    '''
    CHP = CompHaloProp(lims,bins)
    rpmax = rhi.max()
    rbuff=rpmax*search_radius

    x1=0.; x2=box; y1=0.; y2=box; z1=0.; z2=box;
    dx, dy, dz = (x2-x1)/ntile, (y2-y1)/ntile, (z2-z1)/ntile;
    #this stuff is all important for splitting into the tiles
    x1h=it*dx; 
    x2h=(it+1)*dx
    y1h=jt*dy; y2h=(jt+1)*dy
    z1h=kt*dz; z2h=(kt+1)*dz

    x1p=x1h-rbuff; x2p=x2h+rbuff
    y1p=y1h-rbuff; y2p=y2h+rbuff
    z1p=z1h-rbuff; z2p=z2h+rbuff

    xp = pospi[:,0]; yp = pospi[:,1]; zp = pospi[:,2]
    xh = poshi[:,0]; yh = poshi[:,1]; zh = poshi[:,2]

    dmp = [(xp>x1p) & (xp<x2p) & (yp>y1p) & (yp<y2p) & (zp>z1p) & (zp<z2p)]

    
    if mass_kind =='stellar':
        dmh = [(xh>x1h) & (xh<x2h) & (yh>y1h) & (yh<y2h) & (zh>z1h) & (zh<z2h) & (mstari>mhmin) & (mstari<mhmax)]
    elif mass_kind =='halo':
        dmh = [(xh>x1h) & (xh<x2h) & (yh>y1h) & (yh<y2h) & (zh>z1h) & (zh<z2h) & (mhi>mhmin) & (mhi<mhmax)]

    dmh=np.array(dmh[0])
    dmp=np.array(dmp[0])

    posp  = np.column_stack([xp[dmp], yp[dmp], zp[dmp]])
    posh  = np.column_stack([xh[dmp], yh[dmp], zh[dmp]])
    vals          = valsi[:,dmp] #[dmp] #here
    weights       = weightsi[:,dmp] #here

    rh            = rhi[dmh] 
    haloprops = {prop: halosprops[prop][dmh] for prop in halosprops.keys()}

    pcen = np.empty((0),float)
    pval = np.empty((len(volweight),0),float) #here
    pnum = np.empty((len(volweight),0),float) #here

    nhalos=np.shape(xh)[0]
    if rank==0: #when MPI isn't initialized
        print it*ntile**2+jt*ntile+kt+1,'of',ntile**3,'done, nhalos =',nhalos
    
    if nhalos == 0:
        return pcen, pval, pnum, nhalos, haloprops
    
    ninhalos=0
    nphalo = np.zeros(nhalos)
    #weights = 1.0 + 0*xp #now each profile has its own weights
#    posp, vals, weights = precull(posp,vals,weights,posh,rh)

    for ih in np.arange(nhalos):
            pospc, valsc, weightsc = cull_and_center(posp,vals,weights,posh[ih],rpmax,scaled_radius=scaled_radius, search_radius=search_radius, box=box)
            scale=rh[ih]
            pcenc, pvalc, pnumc = CHP.ComputeHaloProfile(pospc,valsc,weightsc,scale,volweight=volweight,scaled_radius=scaled_radius)
            pcen = np.append(pcen,pcenc)
            pval = np.append(pval,pvalc,axis=1) #here
            pnum = np.append(pnum,pnumc,axis=1) #here
    return pcen,pval,pnum,nhalos, haloprops
	
def stackonhalos(
        np.ndarray          posp,
        np.ndarray          vals,
        np.ndarray     volweight,
        np.ndarray       weights,
        np.ndarray          posh,
        np.ndarray            mh,
        np.ndarray            rh,
        np.ndarray            mstar,
        dict haloprops,
        ntile, mhmin, mhmax, scaled_radius, mass_kind, search_radius,lims, bins, box, rank):

    rbuff = rh.max()*search_radius
    print(f"rh max={rh.max():.2f}, search radius={search_radius:.2f}, rbuff={rbuff:.2f}")
    posp,vals,weights = add_ghost_particles(posp,vals,weights,rbuff, box)
    
    #these should all be the same for each prof except pval
    # pcen = np.empty((0),float)
    # pval = np.empty((vals.shape[0],0),float) #here
    # pnum = np.empty((vals.shape[0],0),float) #here

    #     halopropspr={prop: np.empty((0), float) if len(haloprops[prop].shape)==1 else np.empty((0, haloprops[prop].shape[1]), float) for prop in haloprops.keys()}


    pcen_list, pval_list, pnum_list = [], [], []
    halopropspr = {prop: [] for prop in haloprops}

    
    nhalos=0
    
    for it in np.arange(ntile):
        for jt in np.arange(ntile):
            for kt in np.arange(ntile):

                pcenc, pvalc, pnumc, nhalosc, halopropsc= stackonhalostile(posp,vals,posh,volweight,weights,mh,rh,mstar,haloprops,it,jt,kt,ntile,mhmin,mhmax,scaled_radius,mass_kind, lims, bins, search_radius, box, rank)   
                
                # pcen=np.append(pcen,pcenc)
                # pval=np.append(pval,pvalc,axis=1) #here
                # pnum=np.append(pnum,pnumc,axis=1) #here
                
                # for prop in halopropspr.keys():
                #     halopropspr[prop] = np.append(halopropspr[prop], halopropsc[prop], axis=0)

                pcen_list.append(pcenc)
                pval_list.append(pvalc)
                pnum_list.append(pnumc)

                for prop in haloprops:
                    halopropspr[prop].append(halopropsc[prop])
                
                nhalos += nhalosc

    pcen = np.concatenate(pcen_list) if pcen_list else np.empty((0,))
    pval = np.concatenate(pval_list, axis=1) if pval_list else np.empty((vals.shape[0], 0))
    pnum = np.concatenate(pnum_list, axis=1) if pnum_list else np.empty((vals.shape[0], 0))

    for prop in haloprops:
        if halopropspr[prop]:
            halopropspr[prop] = np.concatenate(halopropspr[prop], axis=0)
        else:
            shape = haloprops[prop].shape[1:] if len(haloprops[prop].shape) > 1 else ()
            halopropspr[prop] = np.empty((0,) + shape)
            
    return pcen, pval, pnum, nhalos, halopropspr