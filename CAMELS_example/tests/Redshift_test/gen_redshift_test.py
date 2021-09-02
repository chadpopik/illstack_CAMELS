import numpy as np
import subprocess
import time
import h5py

suite='IllustrisTNG' #IllustrisTNG,SIMBA
        
    
prof1='gasdens'
prof2='gaspth'
prof4='metals_gmw' #gas mass weighted
prof7='gastemp_gmw'


red_dict={'000':6.0,'001':5.0,'002':4.0,'003':3.5,'004':3.0,'005':2.81329,'006':2.63529,'007':2.46560,'008':2.30383,'009':2.14961,'010':2.00259,'011':1.86243,'012':1.72882,'013':1.60144,'014':1.48001,'015':1.36424,'016':1.25388,'017':1.14868,'018':1.04838,'019':0.95276,'020':0.86161,'021':0.77471,'022':0.69187,'023':0.61290,'024':0.53761,'025':0.46584,'026':0.39741,'027':0.33218,'028':0.27,'029':0.21072,'030':0.15420,'031':0.10033,'032':0.04896,'033':0.0}


snap=['033']

#nums=np.linspace(22,32,11,dtype='int')
#simulations=['1P_'+str(i) for i in nums]
    

simulations=['1P_22']
#z=0.04852



start_tot=time.time()
for j in simulations:
    for k in snap:
        file='/home/jovyan/Simulations/'+suite+'/'+j+'/snap_'+k+'.hdf5'
        b=h5py.File(file,'r')
        z=b['/Header'].attrs[u'Redshift']
        
        f=open('params_bins_test.py','r')
        lines=f.readlines()
        lines[0]="basepath = '/home/jovyan/Simulations/"+suite+"/"+j+"/'\n"
        lines[1]="z = %.5f \n"%z
        f.close()
    
        f=open('params_bins_test.py','w')
        f.writelines(lines)
        f.close()
    
        #update the basepath for each sim/snap, clunky but works
        f=open('params_bins_test.py','r')
        lines=f.readlines()
        line0=lines[0]
        line1=lines[1]
        basepath=line0[12:-2]
        print("Basepath check:",basepath)
        print("Redshift check:",line1)    

        g='getprof_temp_profiles_test.sh'
        print('#!/bin/bash',file=open(g,'w'))   
        print('python', 'profiles_bins.py',prof1,prof2,prof4,prof7,k,suite,j,file=open(g,'a'))
        start=time.time()
        subprocess.call(['./getprof_temp_profiles_test.sh'],shell=True)
        end=time.time()
        print("Time elapsed for this sim/snap:",(end-start)/60.,"minutes")
        
        
end_tot=time.time()
print("Total time elapsed:",(end_tot-start_tot)/60.,"minutes")

