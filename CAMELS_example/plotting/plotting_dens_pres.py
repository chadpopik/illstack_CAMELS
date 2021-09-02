import matplotlib.pyplot as plt
import numpy             as np
import h5py

home='/home/jovyan/home/illstack/CAMELS_example/'
 

#-----------------------------------input section
#what suite?
suite='IllustrisTNG'

nums=np.linspace(22,32,11,dtype='int')
simulations=['1P_'+str(i) for i in nums]

ASN1=np.array([0.25,0.33,0.44,0.57,0.76,1.0,1.3,1.7,2.3,3.0,4.0])
#AAGN1=ASN1
#ASN2=np.array([0.5,0.57,0.66,0.76,0.87,1.0,1.15,1.32,1.52,1.74,2.0])
#AAGN2=ASN2


#what redshifts? either enter snapshot as string, or z as array of floats
snap=['033']
#what masses?
mh_low_arr=[10**11.0]
mh_high_arr=[10**12.0]
mh_low_pow=['11']
mh_high_pow=['12']

colors=['r','orchid','orange','gold','lime','green','turquoise','b','navy','blueviolet','k']
#snap=red_dict_tng.keys()  
cut_color=0.6 
mh_cut=True

fig,axes=plt.subplots(1,2,figsize=(12,5)) #20,5
ax1=axes[0]
ax2=axes[1]
ax_arr=[ax1,ax2]

#------------------------------------------------------------


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
    arr=np.array(arr)
    percent_84=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],84),0,arr)
    percent_50=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],50),0,arr)
    percent_16=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],16),0,arr)
    errup=percent_84-percent_50
    errlow=percent_50-percent_16

    std_arr=[]
    for i in range(arr.shape[1]): #for every radial bin
        std_arr.append(np.std(np.apply_along_axis(lambda v: np.log10(v[np.nonzero(v)]),0,arr[:,i])))
    std=np.array(std_arr)
    return errup,errlow,std

def extract(simulation,snap): #extract the quantities,adjust as necessary
    file='/home/jovyan/Simulations/'+suite+'/'+simulation+'/snap_'+snap+'.hdf5'
    b=h5py.File(file,'r')
    z=b['/Header'].attrs[u'Redshift']
    
    stacks=np.load(home+'Batch_NPZ_files_with_CM/'+suite+'/'+suite+'_'+simulation+'_'+snap+'.npz',allow_pickle=True)
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
    
#---------------------------------------------------------------    

for i in np.arange(len(mh_low_arr)):
    mh_low=mh_low_arr[i]
    mh_high=mh_high_arr[i]

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
    
            r_mpc=r/1.e3

            r_mpc_cut,val_dens=outer_cut_multi(20,r_mpc,val_dens)
            r_mpc_cut2,val_dens=inner_cut_multi(3.e-4,r_mpc_cut,val_dens)           
            r_mpc_cut,val_pres=outer_cut_multi(20,r_mpc,val_pres)
            r_mpc_cut2,val_pres=inner_cut_multi(3.e-4,r_mpc_cut,val_pres) 
            
    
            mean_unnorm_dens=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_dens)
            mean_unnorm_pres=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_pres)
            median_unnorm_dens=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_dens)
            median_unnorm_pres=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_pres)
            
            
            label2='%s'%str(ASN1[j])
            
            ax1.loglog(r_mpc_cut2,median_unnorm_dens,color=colors[j])
            ax2.loglog(r_mpc_cut2,median_unnorm_pres,color=colors[j],label=label2)
            #ax1.loglog(r_mpc_cut2,median_unnorm_dens,color=colors[j],linestyle='dashed')
            #ax2.loglog(r_mpc_cut2,median_unnorm_pres,color=colors[j],linestyle='dashed')
            ax1.set_title(r"Mass $%s \leq \log_{10}(M_\odot) \leq %s$"%(mh_low_pow[i],mh_high_pow[i]),size=14)
            ax2.set_title(r"Vary $A_{SN1}$ (Galactic winds)")
            

plt.suptitle(r'$\Omega_m = 0.3, \sigma_8 = 0.8$, z = %.2f'%z)
ax1.set_ylabel(r"$\rho_{gas}(Msol/kpc^3)$",size=14)
ax2.set_ylabel(r"$P_{th} (Msol/kpc/s^2)$",size=14)
ax1.set_xlabel(r'R (Mpc)',size=12)
ax2.set_xlabel(r'R (Mpc)',size=12)

ax2.legend()

plt.subplots_adjust(wspace=0.2,hspace=0.1)
plt.savefig(home+'Figures/'+suite+'_ASN1_z%.2f_median.png'%z)

    