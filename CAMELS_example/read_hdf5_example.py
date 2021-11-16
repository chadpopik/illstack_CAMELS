import matplotlib.pyplot as plt
import numpy             as np
import profile_functions
import h5py

home='/home/jovyan/home/illstack/CAMELS_example/'
 

#-----------------------------------input section
suite='SIMBA'
  
sim='1P_0'
snap='024'
#--------------------------------------------------------------- 

def extract(simulation,snap):
    file='/home/jovyan/Simulations/'+suite+'/'+simulation+'/snap_'+snap+'.hdf5'
    b=h5py.File(file,'r')
    z=b['/Header'].attrs[u'Redshift']
    stacks=h5py.File(home+'Batch_hdf5_files/'+suite+'_'+simulation+'_'+snap+'.hdf5','r')
    val            = stacks['Profiles']
    val_dens       = np.array(val[0,:,:])
    val_pres       = np.array(val[1,:,:])
    bins           = np.array(stacks['nbins'])
    r              = np.array(stacks['r'])
    nprofs         = np.array(stacks['nprofs'])
    mh             = np.array(stacks['Group_M_Crit200']) #units 1e10 Msol/h, M200c
    rh             = np.array(stacks['Group_R_Crit200']) #R200c
    return z,val_dens,bins,r,val_pres,nprofs,mh,rh


z,val_dens,bins,r,val_pres,nprofs,mh,rh=extract(sim,snap)
  
r_mpc=r/1.e3
dens_mean=np.mean(val_dens,axis=0)
     

