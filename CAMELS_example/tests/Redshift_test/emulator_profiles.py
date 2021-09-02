import matplotlib.pyplot as plt
import numpy             as np

file_new='/home/jovyan/home/illstack/CAMELS_example/Tests/Redshift_test/IllustrisTNG_1P_22_0.0.npz'
z_new=0.0 #0.04852

file_old='/home/jovyan/home/illstack/CAMELS_example/Batch_NPZ_files/IllustrisTNG/IllustrisTNG_1P_22_0.0.npz'
z_old=0.0 #0.04896

mh_low=10**12.
mh_high=10**12.2
mh_cut=True

def extract(file):
    
    stacks=np.load(file,allow_pickle=True)
    val            = stacks['val']
    val_dens       = val[0,:]
    val_pres       = val[1,:]
    bins           = stacks['nbins']
    r              = stacks['r']
    nprofs         = stacks['nprofs']
    mh             = stacks['M_Crit200'] #units 1e10 Msol/h, M200c
    rh             = stacks['R_Crit200'] #R200c
    return val_dens,bins,r,val_pres,nprofs,mh,rh

def mhalo_cut(mh_low,mh_high,mh,rh,val_pres,val_dens,bins):
    idx=np.where((mh > mh_low) & (mh < mh_high))
    mh,rh=mh[idx],rh[idx]
    nprofs=len(mh)
    val_pres,val_dens=val_pres[idx,:],val_dens[idx,:]
    val_pres,val_dens=np.reshape(val_pres,(nprofs,bins)),np.reshape(val_dens,(nprofs,bins))
    return mh,rh,val_pres,val_dens,nprofs

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

def correct(z,h,mh,rh,val_dens,val_pres,r):
    comoving_factor=1.0+z
    mh       *= 1e10
    mh       /= h
    rh       /= h
    rh      /= comoving_factor
    val_dens *= 1e10 * h**2
    val_pres *= 1e10 * h**2
    val_pres /= (3.086e16*3.086e16)
    val_dens *= comoving_factor**3
    val_pres *= comoving_factor**3
    r /= h
    r /= comoving_factor
    return mh,rh,val_dens,val_pres,r

h=0.6711
val_dens_new,bins,r_new,val_pres_new,nprofs,mh_new,rh_new=extract(file_new)
mh_new,rh_new,val_dens_new,val_pres_new,r_new=correct(z_new,h,mh_new,rh_new,val_dens_new,val_pres_new,r_new)
mh_new,rh_new,val_pres_new,val_dens_new,nprofs=mhalo_cut(mh_low,mh_high,mh_new,rh_new,val_pres_new,val_dens_new,bins)
r_mpc_new=r_new/1.e3
r_mpc_cut_new,val_dens_new=outer_cut_multi(20,r_mpc_new,val_dens_new)
r_mpc_cut2_new,val_dens_new=inner_cut_multi(0.01,r_mpc_cut_new,val_dens_new)                       
mean_unnorm_dens_new=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_dens_new)



val_dens_old,bins,r_old,val_pres_old,nprofs,mh_old,rh_old=extract(file_old)
mh_old,rh_old,val_dens_old,val_pres_old,r_old=correct(z_old,h,mh_old,rh_old,val_dens_old,val_pres_old,r_old)
mh_old,rh_old,val_pres_old,val_dens_old,nprofs=mhalo_cut(mh_low,mh_high,mh_old,rh_old,val_pres_old,val_dens_old,bins)
r_mpc_old=r_old/1.e3
r_mpc_cut_old,val_dens_old=outer_cut_multi(20,r_mpc_old,val_dens_old)
r_mpc_cut2_old,val_dens_old=inner_cut_multi(0.01,r_mpc_cut_old,val_dens_old)                       
mean_unnorm_dens_old=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_dens_old)


#this is just using the old files, but correcting with the new redshift
val_dens_test,bins,r_test,val_pres_test,nprofs,mh_test,rh_test=extract(file_old)
mh_test,rh_test,val_dens_test,val_pres_test,r_test=correct(z_new,h,mh_test,rh_test,val_dens_test,val_pres_test,r_test)
mh_test,rh_test,val_pres_test,val_dens_test,nprofs=mhalo_cut(mh_low,mh_high,mh_test,rh_test,val_pres_test,val_dens_test,bins)
r_mpc_test=r_test/1.e3
r_mpc_cut_test,val_dens_test=outer_cut_multi(20,r_mpc_test,val_dens_test)
r_mpc_cut2_test,val_dens_test=inner_cut_multi(0.01,r_mpc_cut_test,val_dens_test)                       
mean_unnorm_dens_test=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_dens_test)


print("old",mean_unnorm_dens_old)
print("new",mean_unnorm_dens_new) #this is what we are trying to derive from the dens_old. We have actually run it through illstack so we have the right answer, but we don't want to do that for everything so we need some way to convert.
#the profiles aren't the same, which we also expect due to the comoving factors.
print("old corrected with new z",mean_unnorm_dens_test) #This doesn't work, gives same array as the one using the conversion factor.

#result: the x arrays are the same, which is what we expect
#density with z=0.04852: [4.36950463e+04 1.82625046e+04 5.57812552e+03 1.48476933e+03 5.61001685e+02 2.31207139e+02 9.18023171e+01 4.01347619e+01 2.10466203e+01 1.19755871e+01 9.48712804e+00 7.20123232e+00 4.59767471e+00]

#density with z=0.04896: [4.37221720e+04 1.82680707e+04 5.58221547e+03 1.48531393e+03 5.61020462e+02 2.31433899e+02 9.18390201e+01 4.01614688e+01 2.10628530e+01 1.19870923e+01 9.49760834e+00 7.20837321e+00 4.59582139e+00]

#percent differences: 100*(new-old)/new
#[-0.06207958, -0.03047829, -0.07332123, -0.0366791 , -0.00334705,-0.09807656, -0.03998047, -0.06654306, -0.07712735, -0.09607212, -0.11046863, -0.09916206,  0.04030994]


conversion=((1.+z_new)/(1.+z_old))**3.
print(conversion)
test_new=mean_unnorm_dens_old*conversion
print("old corrected with conversion factor",test_new)

print(mean_unnorm_dens_new/mean_unnorm_dens_old)


    
