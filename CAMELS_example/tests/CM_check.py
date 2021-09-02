import sys
import numpy as np
sys.path.insert(0,'/home/jovyan/home/illstack/')
import illstack as istk
import mpi4py.rc
import matplotlib.pyplot as plt

home='/home/jovyan/home/illstack/CAMELS_example/'
suite='IllustrisTNG'
sim='1P_22'
snap='032'
z=0.04852
file1=np.load(home+'Batch_NPZ_files_with_ID/'+suite+'/'+suite+'_'+sim+'_'+snap+'.npz',allow_pickle=True)
mh1=file1['M_Crit200']
mh1*=1e10
val1=file1['val']
val_dens1=val1[0,:]

file2=np.load(home+'/Batch_NPZ_files_with_CM/'+suite+'/'+suite+'_'+sim+'_'+snap+'.npz',allow_pickle=True)
mh2=file2['M_Crit200']
mh2*=1e10
val2=file2['val']
val_dens2=val2[0,:]
CMx_istk=file2['GroupCMx']



#get IDs
field_list=['Group_M_Crit200','GroupPos','GroupCM']
halos=istk.io.gethalos(int(snap),field_list)
posh=halos['GroupPos']
mh=halos['Group_M_Crit200']
mh*=1e10
CM=halos['GroupCM']
CMx=CM[:,0]

#idx=np.where((mh>=10**11.)&(mh<=10**15.))
#idx=np.array(idx[0])
#mh=mh[idx]
#CMx=CMx[idx]
print("CMx from illstack file",len(CMx_istk),CMx_istk[:10])
#print("CMx from direct group catalog",len(CMx),CMx[:10])


ntile=3
box=25000
mhmin=10**11.
mhmax=10**15.

CMx_ordered=[]
for it in range(ntile):
    for jt in range(ntile):
        for kt in range(ntile):
            xh,yh,zh=posh[:,0],posh[:,1],posh[:,2]
            x1,y1,z1=0.,0.,0.,
            x2,y2,z2=box,box,box
            dx=(x2-x1)/ntile
            dy=(y2-y1)/ntile
            dz=(z2-z1)/ntile
            
            x1h,x2h=it*dx,(it+1)*dx
            y1h,y2h=jt*dy,(jt+1)*dy
            z1h,z2h=kt*dz,(kt+1)*dz

            dmh= [(xh>x1h) & (xh<x2h) & (yh>y1h) & (yh<y2h) & (zh>z1h) & (zh<z2h) & (mh>mhmin) & (mh<mhmax)]
            dmh=np.array(dmh[0])
            xh,yh,zh,CM=xh[dmh],yh[dmh],zh[dmh],CMx[dmh]
            
            for c in CM:
                CMx_ordered.append(c)

print("CMx from illstack process",len(CMx_ordered),CMx_ordered[:10])



