import matplotlib.pyplot as plt
import numpy             as np

home='/home/jovyan/home/illstack/CAMELS_example/'
 

#-----------------------------------input section
#what suite?
suite='IllustrisTNG'
  
nums=np.linspace(22,32,11,dtype='int') #22,65,44 for all
simulations=[]
for n in nums:
    simulations.append('1P_'+str(n))

#what redshifts? either enter snapshot as string, or z as array of floats
snap=['024']
#what masses?
mh_low=10**11. #these are roughly the CMASS limits
mh_high=10**15.
mh_cut=True

#------------------------------------------------------------
red_dict={'000':6.0,'001':5.0,'002':4.0,'003':3.5,'004':3.0,'005':2.81329,'006':2.63529,'007':2.46560,'008':2.30383,'009':2.14961,'010':2.00259,'011':1.86243,'012':1.72882,'013':1.60144,'014':1.48001,'015':1.36424,'016':1.25388,'017':1.14868,'018':1.04838,'019':0.95276,'020':0.86161,'021':0.77471,'022':0.69187,'023':0.61290,'024':0.53761,'025':0.46584,'026':0.39741,'027':0.33218,'028':0.27,'029':0.21072,'030':0.15420,'031':0.10033,'032':0.04896,'033':0.0}


def mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal_gmw,val_temp_gmw,bins):
    idx=np.where((mh > mh_low) & (mh < mh_high))
    mstar,mh,rh,sfr,GroupFirstSub=mstar[idx],mh[idx],rh[idx],sfr[idx],GroupFirstSub[idx]
    nprofs=len(mh)
    val_pres,val_dens,val_metal_gmw,val_temp_gmw=val_pres[idx,:],val_dens[idx,:],val_metal_gmw[idx,:],val_temp_gmw[idx,:]
    val_pres,val_dens,val_metal_gmw,val_temp_gmw=np.reshape(val_pres,(nprofs,bins)),np.reshape(val_dens,(nprofs,bins)),np.reshape(val_metal_gmw,(nprofs,bins)),np.reshape(val_temp_gmw,(nprofs,bins))
    return mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs,val_metal_gmw,val_temp_gmw
    
def outer_cut_multi(outer_cut,x,arr):
    idx=np.where(x <= outer_cut)
    idx=np.array(idx[0])
    x,arr=x[idx],arr[:,idx]
    return x,arr

def inner_cut_multi(inner_cut,x,arr):
    idx=np.where(x >= inner_cut)
    idx=np.array(idx[0])
    x,arr=x[idx],arr[:,idx]
    return x,arr

def get_errors(arr):
    arr=np.array(arr,dtype='float')
    percent_84=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],84),0,arr)
    percent_50=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],50),0,arr)
    percent_16=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],16),0,arr)
    errup=percent_84-percent_50
    errlow=percent_50-percent_16

    std_arr=[]
    for i in range(arr.shape[1]): #for every radial bin
        std_arr.append(np.std(np.apply_along_axis(lambda v: np.log10(v[np.nonzero(v)]),0,arr[:,i])))
    std=np.array(std_arr,dtype='float')
    return errup,errlow,std

def extract(simulation,snap): #extract the quantities,adjust as necessary
    z=red_dict[snap] 
    stacks=np.load(home+'Batch_NPZ_files/'+suite+'/'+suite+'_'+simulation+'_'+str(z)+'.npz',allow_pickle=True)
    val            = stacks['val']
    val_dens       = val[0,:]
    val_pres       = val[1,:]
    val_metal_gmw  = val[2,:]
    val_temp_gmw   = val[3,:]
    bins           = stacks['nbins']
    r              = stacks['r']
    nprofs         = stacks['nprofs']
    mh             = stacks['M_Crit200'] #units 1e10 Msol/h, M200c
    rh             = stacks['R_Crit200'] #R200c
    GroupFirstSub  = stacks['GroupFirstSub']
    sfr            = stacks['sfr'] #Msol/yr
    mstar          = stacks['mstar'] #1e10 Msol/h
    return z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_gmw,val_temp_gmw

def correct(z,h,mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw): #correct all h and comoving factors
    comoving_factor=1.0+z
    mh       *= 1e10
    mstar    *= 1e10
    mh       /= h
    mstar    /= h
    rh       /= h
    rh      /= comoving_factor
    val_dens *= 1e10 * h**2
    val_pres *= 1e10 * h**2
    val_pres /= (3.086e16*3.086e16)
    val_dens *= comoving_factor**3
    val_pres *= comoving_factor**3
    val_temp_gmw*= 1.e10
    #for unscaled
    r /= h
    r /= comoving_factor
    return mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw

def normalize_pressure(nprofs,rh,r,mh,rhocrit_z,omegab,omegam,val_pres):
    G=6.67e-11*1.989e30/((3.086e19)**3) #G in units kpc^3/(Msol*s^2)
    x_values=[]
    norm_pres=[]
    for n in np.arange(nprofs):
        #r200c=(3./4./np.pi/rhombar*mh[i]/200)**(1./3.)
        r200c=rh[n]
        x_values.append(r/r200c)
        P200c=200.*G*mh[n]*rhocrit_z*omegab/(omegam*2.*r200c)
        pressure=val_pres[n,:]
        pressure_divnorm=pressure/P200c
        norm_pres.append(pressure_divnorm)
    mean_xvals=np.mean(x_values, axis=0)
    return mean_xvals,np.array(norm_pres)

def mass_distribution_weight(mh,val_dens,val_pres): #weight by CMASS mass distribution
    logm=np.log10(mh)
    val_dens=np.array(val_dens,dtype='float')
    val_pres=np.array(val_pres,dtype='float')
    b_edges=np.array([12.11179316,12.46636941,12.91135125,13.42362312,13.98474899])
    b_cen=np.array([12.27689266, 12.67884686, 13.16053855, 13.69871423])
    p=np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])
    w=[]
    for i in range(len(mh)):
        index=np.searchsorted(b_edges,logm[i])
        w.append(p[index-1])
    w=np.array(w,dtype='float')

    pres_w=np.apply_along_axis(lambda v: np.average(v[np.nonzero(v)],weights=w[np.nonzero(v)]),0,val_pres)
    dens_w=np.apply_along_axis(lambda v: np.average(v[np.nonzero(v)],weights=w[np.nonzero(v)]),0,val_dens)
    mh=np.average(mh,weights=w)
    return mh,dens_w,pres_w

#--------------------------------------------------------------- 
mean_masses_uw={}
mean_masses_w={}
median_masses={}
for j in np.arange(len(simulations)):
    for k in np.arange(len(snap)):
        z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_gmw,val_temp_gmw=extract(simulations[j],snap[k])
        omegab=0.049
        h=0.6711
        omegam,sigma8=np.loadtxt('/home/jovyan/Simulations/'+suite+'/'+simulations[j]+'/CosmoAstro_params.txt',usecols=(1,2),unpack=True)
        omegalam=1.0-omegam
        rhocrit=2.775e2
        rhocrit_z=rhocrit*(omegam*(1+z)**3+omegalam)
            
        mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw=correct(z,h,mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw)
    
        if mh_cut==True:
                mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs,val_metal_gmw,val_temp_gmw=mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal_gmw,val_temp_gmw,bins)
        plt.figure()
        plt.hist(np.log10(mh))
        plt.xlabel('Mh')
        plt.savefig('mass_histo_'+suite+'_'+simulations[j]+'_'+snap[k]+'.png')
        plt.close()
            
    