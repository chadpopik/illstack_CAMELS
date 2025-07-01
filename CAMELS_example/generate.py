import numpy as np
import subprocess
import time
import h5py

suite='IllustrisTNG'  #IllustrisTNG,SIMBA
suiteset='1P'  # 1P, LH, zoom

#adjust for which batch of simulations
nums=np.arange(51,66, dtype='int')
simulations=[f"{n}" for n in nums if n not in [5, 16, 27, 38, 49, 60]]

#snap=red_dict.keys() #for all snaps
snaps=['024']

profs=['gasdens','gaspth']



savepath = f"/home/jovyan/home/illstack_CAMELS/CAMELS_example/Batch_hdf5_files/{suite}/{suiteset}/"
for sim in simulations:
    # simbasepath = f"/home/jovyan/Data/Sims/{suite}/{suiteset}/{suiteset}_{sim}/"
    simbasepath = f"/home/jovyan/OLD1P/Sims/{suite}/{suiteset}/{suiteset}_{sim}/"
    for snap in snaps:
        # snapfile=h5py.File(simbasepath+f"snapshot_{snap}.hdf5",'r')
        snapfile=h5py.File(simbasepath+f"snap_{snap}.hdf5",'r')
        z=snapfile['/Header'].attrs[u'Redshift']
        h = snapfile['/Header'].attrs[u'HubbleParam']
        
        paramfile=open('istk_params.py','r')
        lines=paramfile.readlines()
        lines[0]=f"basepath = '{simbasepath}'\n"
        lines[1]=f"z = {z} \n"
        lines[10]=f"save_direct = '{savepath}'\n"
        lines[11]=f"h = {h}"
        paramfile.close()
    
        paramfile=open('istk_params.py','w')
        paramfile.writelines(lines)
        paramfile.close()
    
        #update the basepath for each sim/snap, clunky but works
        f=open('istk_params.py','r')
        lines=f.readlines()
        line0=lines[0]
        line1=lines[1]
        basepath=line0[12:-2]
        print("Basepath check:",basepath)
        print("Redshift check:",line1)
    

        g='getprof_temp_profiles.sh'
        print('#!/bin/bash',file=open(g,'w'))   
        print('python', 'profiles_expand.py',suite,suiteset,sim,snap,*profs,file=open(g,'a'))
        start=time.time()
        subprocess.call(['./getprof_temp_profiles.sh'],shell=True)
        
end=time.time()
print("Time elapsed:",(end-start)/60.,"minutes")

