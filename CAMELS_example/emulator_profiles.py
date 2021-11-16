import matplotlib.pyplot as plt
import numpy             as np
import profile_functions
import h5py

home='/home/jovyan/home/illstack/CAMELS_example/'
 

#-----------------------------------input section
suite='IllustrisTNG'
emulator_type='general'  #CMASS, general
#for CMASS emulator use mopc_profiles.py. Gives correct radial range. Can still use this script to make plots though.
  
nums=np.linspace(22,65,44,dtype='int') #0,65,66 for all. 22,65,44
simulations=['1P_'+str(n) for n in nums]

if emulator_type=='general':
    snap_arr=['033','032','031','030','029','028','027','026','025','024']
    mass_str_arr=['11-11.5','11.5-12','12-12.3','12.3-13.1']
    mh_low_arr=[10**11.,10**11.5+0.1,10**12.+0.1,10**12.3+0.1]
    mh_high_arr=[10**11.5,10**12.,10**12.3,10**13.1]
    mh_low_pow_arr=[11,11.5,12,12.3]
    mh_high_pow_arr=[11.5,12,12.3,13.1]

elif emulator_type=='CMASS':
    snap_arr=['024']
    mass_str_arr=['12.12-13.2'] #13.98, there are no halos above 13.1
    mh_low_arr=[10**12.12]
    mh_high_arr=[10**13.2] #13.98
    mh_low_pow_arr=[12.12]
    mh_high_pow_arr=[13.2]

#--------------------------------------------------------------- 

def extract(simulation,snap): #extract the quantities,adjust as necessary
    file='/home/jovyan/Simulations/'+suite+'/'+simulation+'/snap_'+snap+'.hdf5'
    b=h5py.File(file,'r')
    z=b['/Header'].attrs[u'Redshift']
    
    stacks=np.load(home+'Batch_NPZ_files_with_CM/'+suite+'/'+suite+'_'+simulation+'_'+snap+'.npz'%z,allow_pickle=True)
    val            = stacks['Profiles']
    val_dens       = val[0,:]
    val_pres       = val[1,:]
    val_metal_gmw  = val[2,:]
    val_temp_gmw   = val[3,:]
    bins           = stacks['nbins']
    r              = stacks['r']
    nprofs         = stacks['nprofs']
    mh             = stacks['Group_M_Crit200'] #units 1e10 Msol/h, M200c
    rh             = stacks['Group_R_Crit200'] #R200c
    GroupFirstSub  = stacks['GroupFirstSub']
    sfr            = stacks['GroupSFR'] #Msol/yr
    mstar          = stacks['GroupMassType_Stellar'] #1e10 Msol/h
    return z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_gmw,val_temp_gmw

mean_masses_uw={}
mean_masses_w={}
median_masses={}
for j in np.arange(len(simulations)):
    sim=simulations[j]
    for k in np.arange(len(snap_arr)):
        snap=snap_arr[k]
        z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_gmw,val_temp_gmw=extract(sim,snap)
        omegab=0.049
        h=0.6711
        omegam,sigma8=np.loadtxt('/home/jovyan/Simulations/'+suite+'/'+simulations[j]+'/CosmoAstro_params.txt',usecols=(1,2),unpack=True)
        omegalam=1.0-omegam
        rhocrit=2.775e2
        rhocrit_z=rhocrit*(omegam*(1+z)**3+omegalam)
            
        mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw=profile_functions.correct(z,h,mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw)
    
        for m in np.arange(len(mh_low_arr)):
            mh_low=mh_low_arr[m]
            mh_high=mh_high_arr[m]
            mass_str=mass_str_arr[m]
            mh_low_pow=mh_low_pow_arr[m]
            mh_high_pow=mh_high_pow_arr[m]
            print(sim,snap,mass_str)
            mstarm,mhm,rhm,sfrm,GroupFirstSubm,val_presm,val_densm,nprofsm,val_metal_gmwm,val_temp_gmwm=profile_functions.mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal_gmw,val_temp_gmw,bins)
    
            r_mpc=r/1.e3
            
            #outer cut 20, inner cut 2e-3 for TNG, SIM can do 5e-4
            r_mpc_cut,val_densm=profile_functions.outer_cut_multi(20,r_mpc,val_densm)
            r_mpc_cut2,val_densm=profile_functions.inner_cut_multi(2.e-3,r_mpc_cut,val_densm)
            r_mpc_cut,val_presm=profile_functions.outer_cut_multi(20,r_mpc,val_presm)
            r_mpc_cut2,val_presm=profile_functions.inner_cut_multi(2.e-3,r_mpc_cut,val_presm)
            r_mpc_cut,val_metal_gmwm=profile_functions.outer_cut_multi(20,r_mpc,val_metal_gmwm)
            r_mpc_cut2,val_metal_gmwm=profile_functions.inner_cut_multi(2.e-3,r_mpc_cut,val_metal_gmwm)
            r_mpc_cut,val_temp_gmwm=profile_functions.outer_cut_multi(20,r_mpc,val_temp_gmwm)
            r_mpc_cut2,val_temp_gmwm=profile_functions.inner_cut_multi(2.e-3,r_mpc_cut,val_temp_gmwm)
                        
            mean_unnorm_densm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_densm)
            mean_unnorm_presm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_presm)
            median_unnorm_densm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_densm)
            median_unnorm_presm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_presm)
            mean_unnorm_metal_gmwm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_metal_gmwm)
            mean_unnorm_temp_gmwm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_temp_gmwm)
            median_unnorm_metal_gmwm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_metal_gmwm)
            median_unnorm_temp_gmwm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_temp_gmwm)
        
            errup_dens_unnormm,errlow_dens_unnormm,std_dens_unnormm=profile_functions.get_errors(val_densm)
            errup_pres_unnormm,errlow_pres_unnormm,std_pres_unnormm=profile_functions.get_errors(val_presm)
            errup_metal_gmw_unnormm,errlow_metal_gmw_unnormm,std_metal_gmw_unnormm=profile_functions.get_errors(val_metal_gmwm)
            errup_temp_gmw_unnormm,errlow_temp_gmw_unnormm,std_temp_gmw_unnormm=profile_functions.get_errors(val_temp_gmwm)
            mean_masses_uw[sim]=np.mean(mhm)
            median_masses[sim]=np.median(mhm)
        
        
            if emulator_type=='general':
                header='R (Mpc), mean rho (Msol/kpc^3), errup (Msol/kpc^3), errlow, std, median rho (Msol/kpc^3), mean pth (Msol/kpc/s^2), errup(Msol/kpc/s^2), errlow, std, median pth (Msol/kpc/s^2), mean gas-mass-weighted metal (fraction), errup, errlow, std, median metal, mean gas-mass-weighted temp (K), errup, errlow, std, median temp  \n nprofs %i, mean mh %f, median mh %f \n Mass range %.2f - %.1f'%(nprofsm,np.mean(mhm),np.median(mhm),mh_low_pow,mh_high_pow)
                np.savetxt(home+'Emulator_profiles/mass_bins_11-13/'+suite+'/'+suite+'_'+sim+'_'+snap+'_uw_%s.txt'%mass_str,np.c_[r_mpc_cut2,mean_unnorm_densm, errup_dens_unnormm,errlow_dens_unnormm,std_dens_unnormm,median_unnorm_densm,mean_unnorm_presm,errup_pres_unnormm,errlow_pres_unnormm,std_pres_unnormm,median_unnorm_presm,mean_unnorm_metal_gmwm, errup_metal_gmw_unnormm,errlow_metal_gmw_unnormm,std_metal_gmw_unnormm,median_unnorm_metal_gmwm,mean_unnorm_temp_gmwm, errup_temp_gmw_unnormm,errlow_temp_gmw_unnormm,std_temp_gmw_unnormm,median_unnorm_temp_gmwm],header=header)
        
            if emulator_type=='CMASS':
                mean_mh,mean_unnorm_dens_w,mean_unnorm_pres_w=profile_functions.mass_distribution_weight(mhm,val_densm,val_presm)
                header='R (Mpc), rho (Msol/kpc^3), errup, errlow, std, pth (Msol/kpc/s^2), errup, errlow, std \n nprofs %i, average weighted mh %f'%(nprofs,mean_mh)
                mean_masses_w[sim]=mean_mh
                np.savetxt(home+'Emulator_profiles/'+suite+'_'+sim+'_'+snap+'_w.txt',np.c_[r_mpc_cut2,mean_unnorm_dens_w, errup_dens_unnormm,errlow_dens_unnormm,std_dens_unnormm,mean_unnorm_pres_w,errup_pres_unnormm,errlow_pres_unnormm,std_pres_unnormm],header=header)

    
