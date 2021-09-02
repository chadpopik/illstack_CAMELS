import sys
import numpy as np
sys.path.insert(0,'/home/jovyan/home/illstack/')
import illstack as istk
import mpi4py.rc
import matplotlib.pyplot as plt

home='/home/jovyan/home/illstack/CAMELS_example/Batch_NPZ_files_with_ID/'
suite='IllustrisTNG'
sim='1P_22'
snap='032'
z=0.04852
file=np.load(home+suite+'/'+suite+'_'+sim+'_'+str(z)+'.npz',allow_pickle=True)
ID_istk=file['ID']
mh_istk=file['M_Crit200']
mh_istk*=1e10
ID_istk=[int(i) for i in ID_istk]
ID_istk_sorted=sorted(ID_istk)
val=file['val']
val_dens=val[0,:]

#get IDs
field_list=['Group_M_Crit200','Group_R_Crit200','GroupPos']
halos=istk.io.gethalos(int(snap),field_list)
posh=halos['GroupPos']
mh=halos['Group_M_Crit200']
mh*=1e10
#idx=np.where((mh>=10**11.)&(mh<=10**15.))
#idx=np.array(idx[0])
#ID=idx
#mh=mh[idx]

print("IDs from illstack tracking",len(ID_istk),ID_istk)
#print("IDs from direct group catalog",len(ID),ID)

#these are the same
#print(mh[23]), print(mh_istk[0])

ntile=3
box=25000
mhmin=10**11.
mhmax=10**15.
ID_arr=np.linspace(0,len(posh)-1,len(posh))

ID_tot=[]
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
            xh,yh,zh,ID=xh[dmh],yh[dmh],zh[dmh],ID_arr[dmh]
            
            for I in ID:
                ID_tot.append(I)

ID_tot=[int(i) for i in ID_tot]
print(ID_tot)



