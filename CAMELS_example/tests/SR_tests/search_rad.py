#!/usr/bin/env python
import sys
import numpy             as np
sys.path.insert(0,'/home/jovyan/home/illstack/')
import illstack as istk
import params_test as params
import h5py

istk.init.initialize('params_test.py')

snap_num= int(sys.argv[1])
suite=str(sys.argv[2])
simulation=str(sys.argv[3])

z=params.z

#z=red_dict[str(snap_num).zfill(3)]
comoving_factor=1.+z
h=0.677
mlow=10**11.
mhigh=10**15.
mhmin = mlow /1e10 # minimum mass in 1e10 Msun/h
mhmax = mhigh /1e10 # maximum mass in 1e10 Msun/h


field_list = ['Group_M_Crit200','Group_R_Crit200','Group_R_Crit500']

halos = istk.io.gethalos(snap_num,field_list)
rh   = halos['Group_R_Crit200']
mh   = halos['Group_M_Crit200']

#print(len(mh))
idx=np.where((mh >= mhmin) & (mh <= mhmax))
idx=np.array(idx[0])
mh,rh=mh[idx],rh[idx]
#print(len(mh))
#rh /= h
#rh /= comoving_factor

#check the search radius
max_rh=max(rh)/1.e3
SR=10./max_rh

f=open('/home/jovyan/home/illstack/CAMELS_example/Tests/SR_tests/sr_test_'+suite+'_'+str(sys.argv[1])+'.txt','a')
f.write('%s snap %s, need SR %f (max rh %f cMpc/h)\n'%(simulation,snap_num,SR,max_rh))
f.close()

print(simulation,snap_num)



