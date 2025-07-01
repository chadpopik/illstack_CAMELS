import matplotlib.pyplot as plt
import numpy             as np
import h5py


home='/home/jovyan/home/illstack_CAMELS/CAMELS_example/'
 
suite='IllustrisTNG'
ASN1=np.array([0.25,0.33,0.44,0.57,0.76,1.0,1.3,1.7,2.3,3.0,4.0])
ASN2=np.array([0.5,0.57,0.66,0.76,0.87,1.0,1.15,1.32,1.52,1.74,2.0])

snap=['026']
feedback='AAGN1'

if feedback=='ASN1':
    simulations=['1P_22','1P_23','1P_24','1P_25','1P_26','1P_27','1P_28','1P_29','1P_30','1P_31','1P_32']
    vary=ASN1
    title2=r"Vary $A_{SN1}$"
elif feedback=='AAGN1':
    simulations=['1P_33','1P_34','1P_35','1P_36','1P_37','1P_38','1P_39','1P_40','1P_41','1P_42','1P_43']
    vary=ASN1
    title2=r"Vary $A_{AGN1}$"
elif feedback=='ASN2':
    simulations=['1P_44','1P_45','1P_46','1P_47','1P_48','1P_49','1P_50','1P_51','1P_52','1P_53','1P_54']
    vary=ASN2
    title2=r"Vary $A_{SN2}$"
elif feedback=='AAGN2':
    simulations=['1P_55','1P_56','1P_57','1P_58','1P_59','1P_60','1P_61','1P_62','1P_63','1P_64','1P_65']
    vary=ASN2
    title2=r"Vary $A_{AGN2}$"

colors=['r','orchid','orange','gold','lime','green','turquoise','b','navy','blueviolet','k']
mh_low_arr=[10**12.]
mh_high_arr=[10**12.2] 
mh_low_pow=['12']
mh_high_pow=['12.2']
mh_cut=True

#which profiles? just assume all for now, so 2x2
fig,axes=plt.subplots(2,2,figsize=(12,10)) #20,5
ax1=axes[0,0]
ax2=axes[0,1]
ax3=axes[1,0]
ax4=axes[1,1]
ax_arr=[ax1,ax2,ax3,ax4]

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
    z=red_dict[snap] 
    stacks=h5py.File(home+'Batch_hdf5_files/'+suite+'/test/'+suite+'_'+simulation+'_'+str(z)+'.h5', 'r')
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
            omegam,sigma8=np.loadtxt('/home/jovyan/Simulations/IllustrisTNG/'+simulations[j]+'/CosmoAstro_params.txt',usecols=(1,2),unpack=True)
            omegalam=1.0-omegam
            rhocrit=2.775e2
            rhocrit_z=rhocrit*(omegam*(1+z)**3+omegalam)
            
            mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw=correct(z,h,mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw)
    
            if mh_cut==True:
                    mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs,val_metal_gmw,val_temp_gmw=mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal_gmw,val_temp_gmw,bins)
    
            r_mpc=r/1.e3
            
            print("sim",simulations[j])
            r_mpc_cut,val_dens=outer_cut_multi(20,r_mpc,val_dens)
            r_mpc_cut2,val_dens=inner_cut_multi(0.01,r_mpc_cut,val_dens)           
            r_mpc_cut,val_pres=outer_cut_multi(20,r_mpc,val_pres)
            r_mpc_cut2,val_pres=inner_cut_multi(0.01,r_mpc_cut,val_pres)
            r_mpc_cut,val_metal_gmw=outer_cut_multi(20,r_mpc,val_metal_gmw)
            r_mpc_cut2,val_metal_gmw=inner_cut_multi(0.01,r_mpc_cut,val_metal_gmw)
            r_mpc_cut,val_temp_gmw=outer_cut_multi(20,r_mpc,val_temp_gmw)
            r_mpc_cut2,val_temp_gmw=inner_cut_multi(0.01,r_mpc_cut,val_temp_gmw)  
            
    
            mean_unnorm_dens=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_dens)
            mean_unnorm_pres=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_pres)
            mean_unnorm_metal_gmw=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_metal_gmw)
            mean_unnorm_temp_gmw=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_temp_gmw)
            
            median_unnorm_dens=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_dens)
            median_unnorm_pres=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_pres)
            median_unnorm_metal_gmw=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_metal_gmw)
            median_unnorm_temp_gmw=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_temp_gmw)
            
            
            if len(simulations) >1: #if we are varying a param
                label1='%s nhalo:%i'%(simulations[j],nprofs)
                label2='%s'%str(vary[j])
                ax2.set_title(title2)
                index=j
            elif len(snap) >1: #if we are varying redshift
                label1='nhalo:%i'%nprofs
                label2='z = %.2f'%z
                ax2.set_title(title2)
                index=k
            else:
                label1='1P_22'
                label2='simba'
                index=0
            
            ax1.loglog(r_mpc_cut2,mean_unnorm_dens,color=colors[index])
            ax2.loglog(r_mpc_cut2,mean_unnorm_pres,color=colors[index],label=label2)
            ax3.loglog(r_mpc_cut2,mean_unnorm_metal_gmw,color=colors[index])
            ax4.loglog(r_mpc_cut2,mean_unnorm_temp_gmw,color=colors[index])
            
            #ax1.loglog(r_mpc_cut2,median_unnorm_dens,color=colors[index],linestyle='dashed')
            #ax2.loglog(r_mpc_cut2,median_unnorm_pres,color=colors[index],linestyle='dashed',label=label2)
            #ax3.loglog(r_mpc_cut2,median_unnorm_metal_gmw,color=colors[index],linestyle='dashed')
            #ax4.loglog(r_mpc_cut2,median_unnorm_temp_gmw,color=colors[index],linestyle='dashed')
            
            ax1.set_title(r"Mass $%s \leq M_\odot \leq %s, z=%.2f$"%(mh_low_pow[i],mh_high_pow[i],z),size=14)

            

plt.suptitle(r'$\Omega_m = 0.3, \sigma_8 = 0.8$')
ax1.set_ylabel(r"$\rho_{gas}(Msol/kpc^3)$",size=14)
ax2.set_ylabel(r"$P_{th} (Msol/kpc/s^2)$",size=14)
ax3.set_ylabel(r"Metal fraction",size=14)
ax4.set_ylabel(r"Temperature (K)",size=14)
ax3.set_xlabel(r'R (Mpc)',size=12)
ax4.set_xlabel(r'R (Mpc)',size=12)

#ax1.legend()
ax2.legend()
#ax3.legend()
#ax4.legend()

plt.subplots_adjust(wspace=0.2,hspace=0.1)
#plt.savefig(home+'Figures/mean_'+feedback+'_test.png')

    