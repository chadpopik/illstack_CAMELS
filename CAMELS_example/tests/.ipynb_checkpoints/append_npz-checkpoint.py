import sys
import numpy as np
sys.path.insert(0,'/home/jovyan/home/illstack/')
import illstack as istk
sys.path.insert(0,'/home/jovyan')
sys.path.append('/home/jovyan/illustris_python')
import illustris_python as il
import mpi4py.rc

red_dict={'000':6.0,'001':5.0,'002':4.0,'003':3.5,'004':3.0,'005':2.81329,'006':2.63529,'007':2.46560,'008':2.30383,'009':2.14961,'010':2.00259,'011':1.86243,'012':1.72882,'013':1.60144,'014':1.48001,'015':1.36424,'016':1.25388,'017':1.14868,'018':1.04838,'019':0.95276,'020':0.86161,'021':0.77471,'022':0.69187,'023':0.61290,'024':0.53761,'025':0.46584,'026':0.39741,'027':0.33218,'028':0.27,'029':0.21072,'030':0.15420,'031':0.10033,'032':0.04896,'033':0.0}

keys=['r','val','n','M_Crit200','R_Crit200','nprofs','nbins','GroupFirstSub','sfr','mstar','GroupBHMass','GroupBHMdot',',Group_GasH','Group_GasHe','Group_GasC','Group_GasN','Group_GasO','Group_GasNe','Group_GasMg','Group_GasSi','Group_GasFe','GroupGasMetallicity','GroupLen','GroupMass','GroupNsubs','Group_StarH','Group_StarHe','Group_StarC','Group_StarN','Group_StarO','Group_StarNe','Group_StarMg','Group_StarSi','Group_StarFe','GroupStarMetallicity','GroupVelx','GroupVely','GroupVelz','GroupWindMass','M_Crit500','M_Mean200','M_TopHat200','R_Crit500','R_Mean200','R_TopHat200']

home='/home/jovyan/home/illstack/CAMELS_example/'
suite='IllustrisTNG'
nums=np.linspace(22,65,44,dtype='int')
simulations=['1P_'+str(n) for n in nums]
snap_arr=['033','032','031','030','029','028','027','026','025','024']

def get_IDs(sim,snap):
    basepath='/home/jovyan/Simulations/'+suite+'/'+sim+'/'
    field_list=['Group_M_Crit200','GroupPos']
    halos=il.groupcat.loadHalos(basepath,int(snap),field_list)
    posh=halos['GroupPos']
    mh=halos['Group_M_Crit200']
    mh*=1e10
    
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
                    ID_tot.append(int(I))
    return ID_tot

def get_CM(sim,snap):
    basepath='/home/jovyan/Simulations/'+suite+'/'+sim+'/'
    field_list=['Group_M_Crit200','GroupPos','GroupCM']
    halos=il.groupcat.loadHalos(basepath,int(snap),field_list)
    posh=halos['GroupPos']
    mh=halos['Group_M_Crit200']
    mh*=1e10
    GroupCM=halos['GroupCM']
    GroupCMx=GroupCM[:,0]
    GroupCMy=GroupCM[:,1]
    GroupCMz=GroupCM[:,2]
    
    ntile=3
    box=25000
    mhmin=10**11.
    mhmax=10**15.
    
    CMx=[]
    CMy=[]
    CMz=[]
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
                xh,yh,zh,CMxh,CMyh,CMzh=xh[dmh],yh[dmh],zh[dmh],GroupCMx[dmh],GroupCMy[dmh],GroupCMz[dmh]
            
                for c in np.arange(len(CMxh)):
                    CMx.append(CMxh[c])
                    CMy.append(CMyh[c])
                    CMz.append(CMzh[c])
    return CMx,CMy,CMz


for sim in range(len(simulations)):
    for snap in range(len(snap_arr)): 
        print("sim",simulations[sim],"snap",snap_arr[snap])
        
        #these are for adding IDs
        #old_z=red_dict[snap_arr[snap]]
        #file=np.load(home+'Batch_NPZ_files/'+suite+'/'+suite+'_'+simulations[sim]+'_'+str(old_z)+'.npz',allow_pickle=True)
        
        #these are for adding CM
        file=np.load(home+'Batch_NPZ_files_with_ID/'+suite+'/'+suite+'_'+simulations[sim]+'_'+snap_arr[snap]+'.npz',allow_pickle=True)
        r=file['r']
        val=file['val']
        n=file['n']
        M_Crit200=file['M_Crit200']
        R_Crit200=file['R_Crit200']
        nprofs=file['nprofs']
        nbins=file['nbins']
        GroupFirstSub=file['GroupFirstSub']
        sfr=file['sfr']
        mstar=file['mstar']
        GroupBHMass=file['GroupBHMass']
        GroupBHMdot=file['GroupBHMdot']
        Group_GasH=file['Group_GasH']
        Group_GasHe=file['Group_GasHe']
        Group_GasC=file['Group_GasC']
        Group_GasN=file['Group_GasN']
        Group_GasO=file['Group_GasO']
        Group_GasNe=file['Group_GasNe']
        Group_GasMg=file['Group_GasMg']
        Group_GasSi=file['Group_GasSi']
        Group_GasFe=file['Group_GasFe']
        GroupGasMetallicity=file['GroupGasMetallicity']
        GroupLen=file['GroupLen']
        GroupMass=file['GroupMass']
        GroupNsubs=file['GroupNsubs']
        Group_StarH=file['Group_StarH']
        Group_StarHe=file['Group_StarHe']
        Group_StarC=file['Group_StarC']
        Group_StarN=file['Group_StarN']
        Group_StarO=file['Group_StarO']
        Group_StarNe=file['Group_StarNe']
        Group_StarMg=file['Group_StarMg']
        Group_StarSi=file['Group_StarSi']
        Group_StarFe=file['Group_StarFe']
        GroupStarMetallicity=file['GroupStarMetallicity']
        GroupVelx=file['GroupVelx']
        GroupVely=file['GroupVely']
        GroupVelz=file['GroupVelz']
        GroupWindMass=file['GroupWindMass']
        M_Crit500=file['M_Crit500']
        M_Mean200=file['M_Mean200']
        M_TopHat200=file['M_TopHat200']
        R_Crit500=file['R_Crit500']
        R_Mean200=file['R_Mean200']
        R_TopHat200=file['R_TopHat200']
        ID=file['ID']

        #get IDs
        #ID=get_IDs(simulations[sim],int(snap_arr[snap]))
        
        #get CM
        GroupCMx,GroupCMy,GroupCMz=get_CM(simulations[sim],int(snap_arr[snap]))

        np.savez(home+'Batch_NPZ_files_with_CM/'+suite+'/'+suite+'_'+simulations[sim]+'_'+snap_arr[snap]+'.npz',r=r,val=val,n=n,M_Crit200=M_Crit200,R_Crit200=R_Crit200,nprofs=nprofs,nbins=nbins,GroupFirstSub=GroupFirstSub,sfr=sfr,mstar=mstar,GroupBHMass=GroupBHMass,GroupBHMdot=GroupBHMdot,GroupCMx=GroupCMx,GroupCMy=GroupCMy,GroupCMz=GroupCMz,Group_GasH=Group_GasH,Group_GasHe=Group_GasHe,Group_GasC=Group_GasC,Group_GasN=Group_GasN,Group_GasO=Group_GasO,Group_GasNe=Group_GasNe,Group_GasMg=Group_GasMg,Group_GasSi=Group_GasSi,Group_GasFe=Group_GasFe,GroupGasMetallicity=GroupGasMetallicity,GroupLen=GroupLen,GroupMass=GroupMass,GroupNsubs=GroupNsubs,Group_StarH=Group_StarH,Group_StarHe=Group_StarHe,Group_StarC=Group_StarC,Group_StarN=Group_StarN,Group_StarO=Group_StarO,Group_StarNe=Group_StarNe,Group_StarMg=Group_StarMg,Group_StarSi=Group_StarSi,Group_StarFe=Group_StarFe,GroupStarMetallicity=GroupStarMetallicity,GroupVelx=GroupVelx,GroupVely=GroupVely,GroupVelz=GroupVelz,GroupWindMass=GroupWindMass,M_Crit500=M_Crit500,M_Mean200=M_Mean200,M_TopHat200=M_TopHat200,R_Crit500=R_Crit500,R_Mean200=R_Mean200,R_TopHat200=R_TopHat200,ID=ID)






