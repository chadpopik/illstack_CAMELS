"""
This file is responsible for extracting data, and doing things like converting from units, adding radii or mass cuts, and doing weighted average/binning stuff.
"""

import sys
sys.path.append('/home/jovyan/home/')
from CAMELS_stuff.Basics import *
setplot(dark=True)




# General function to convert the units from sims/profiles out of comoving/littleh units, would be good to continue to generalize
def convert(haloprops, z, h, profs={}):
    newhaloprops = {}
    for prop in haloprops.keys():
        if prop in ['Group_M_Crit200', 'GroupMassType_Stellar', 'GroupBHMass', 'GroupMass', 'GroupWindMass', 'Group_M_Crit500', 'Group_M_Mean200', 'Group_M_TopHat200']:  # 1e10 Msol/h to Msun
            newhaloprops[prop] = haloprops[prop][()] *1e10/h
        elif prop in ['Group_R_Crit200', 'Group_R_Crit500', 'Group_R_Mean200','Group_R_TopHat200', 'GroupPos']:  # ckpc/h to Mpc
            newhaloprops[prop] = haloprops[prop][()] /1e3/(1+z)/h
        elif prop in ['GroupVelx', 'GroupVely', 'GroupVelz']:  # km/s/a to km/s
            newhaloprops[prop] = haloprops[prop][()] *(1+z)
        elif prop=='GroupBHMdot': # 1e10 (Msol/h)/(0.978 Gyr/h) to Msun/yr
            newhaloprops[prop] = haloprops[prop][()] *1e10/(0.978*1e9)
        else:
            newhaloprops[prop] = haloprops[prop][()]

    newhaloprofs = {}
    for prof in profs.keys():
        if prof in ['r']:  # ckpc/h to Mpc
            newhaloprofs[prof] = profs[prof][()] /(1+z)/h/1.e3
        elif prof in ['dens', 'gasdens']:  # (1e10*Msun/h)/(kpc/h)^3 to Msun/Mpc^3
            newhaloprofs[prof] = profs[prof][()] *1e10*1e3**3*(1+z)**3*h**2
        elif prof in ['pres', 'gaspth']:  # (1e10*Msun/h)*(km/s)^2/(ckpc/h)**3 to Msun/Mpc/s^2
             newhaloprofs[prof] = profs[prof][()] *1e10*(u.km**2/u.kpc**3).to(1/u.Mpc)*(1+z)**3*h**2
        elif prof in ['temp_gmw']:  # 1e10K to K
            newhaloprofs[prof] = profs[prof][()] *1e10
        else:
            newhaloprofs[prof] = profs[prof][()]

    return newhaloprops, newhaloprofs


def cut(profs, haloprops, mh, r, mh_low=0, mh_high=np.inf, inner_cut=0, outer_cut=np.inf):
    # Cut properties by mass
    mcut = (mh > mh_low) & (mh < mh_high)
    cutprops = {prop: haloprops[prop][mcut] for prop in haloprops.keys()}

    # Cut profiles by mass and radius, also cutting radii bins with no particles
    rcut = (r[mcut] <= outer_cut) & (r[mcut] >= inner_cut) & (np.sum(profs['npart'][mcut], axis=0)!=0)
    cutprofs = {prof: profs[prof][mcut][rcut].reshape(rcut.shape[0], -1) for prof in profs.keys()}
    return cutprofs, cutprops





# # Extract profiles from old ones that Emily ran
# def extract_old(basepath, suite, subset, sim, snap):
#     # The profiles and old 1P sims use a snap out of 34 instead of the 91 ordering, so have to convert
#     zdf = pd.read_csv(f'/home/jovyan/home/illstack_CAMELS/zs.csv')
#     oldsnap = int(zdf.loc[zdf['91 snapshots']==int(snap), '34 snapshots'].iloc[0])

#     if subset=='1P':  # 1P alternates between naming conventions and requires some translating
#         if sim%11-5<0: 
#             subfolder = f'1P_{(np.ceil(sim/10+0.1)):.0f}_n{np.abs(sim%11-5)}'
#         else: 
#             subfolder = f'1P_{(np.ceil(sim/11)):.0f}_{sim%11-5}'
#         if sim%11-5==0:  # The fiducial values aren't in the OLD1P so have to fetch from the new ones
#             simparams = h5py.File(f'{basepath}/Sims/{suite}/{subset}/1P_p{(np.ceil(sim/10)):.0f}_{sim%11-5}/snapshot_{snap:03}.hdf5', 'r')['Header'].attrs
#         else:
#             simparams = h5py.File(f'/home/jovyan/OLD1P/Sims/{suite}/{subset}/{subfolder}/snap_{oldsnap:03}.hdf5', 'r')['Header'].attrs

#     else:
#         subfolder = f'{subset}_{sim}'
#         simparams = h5py.File(f'{basepath}/Sims/{suite}/{subset}/{subfolder}/snapshot_{snap:03}.hdf5', 'r')['Header'].attrs

#     # Load profiles and properties into dictionaries
#     profiles = h5py.File(f'{basepath}/Profiles/{suite}/{subset}/{subfolder}/{suite}_{subset}_{sim}_{oldsnap:03}.hdf5', 'r')
#     haloprops = {key:profiles[key][()] for key in list(profiles.keys()) if profiles[key].shape==profiles['ID'].shape}
#     valprofiles = {'r' : profiles['r'][()]*np.ones(profiles['n'][0].shape),
#                    'npart' : profiles['n'][0],
#                    'dens' : profiles['Profiles'][0], 'pres' : profiles['Profiles'][1],'metal_gmw' : profiles['Profiles'][2],'temp_gmw' : profiles['Profiles'][3]}        

#     return valprofiles, haloprops, dict(simparams)







# for sim in simulations:
#     for snap in snapshots:
#         r, valprofiles, haloprops, simparams = extract(suite, subset, sim, snap)

#         file_hdf5=h5py.File(f"{save_direct}/{suite}_{subset}_{sim}_{snap}.hdf5",'w',track_order=True)
#         for prof in profs.keys():
#             file_hdf5.create_dataset(f'{prof}/Mean', data=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), 0, profs[prof]), dtype='float64')
#             file_hdf5.create_dataset(f'{prof}/STD', data=np.apply_along_axis(lambda v: np.std(v[np.nonzero(v)]), 0, profs[prof]), dtype='float64')
#             file_hdf5.create_dataset(f'{prof}/P50', data=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],50), 0, profs[prof]), dtype='float64')
#             file_hdf5.create_dataset(f'{prof}/P16', data=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],16), 0, profs[prof]), dtype='float64')
#             file_hdf5.create_dataset(f'{prof}/P80', data=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],84), 0, profs[prof]), dtype='float64')

#             if weights are given:
#                 ws = weights[np.searchsorted(bin_edges, np.log10(weightby))-1]
#                 measures['weighted'] = np.apply_along_axis(lambda v: np.average(v[np.nonzero(v)], weights=ws[np.nonzero(v)]), 0, prof)

#         b_edges=np.array([12.11179316,12.46636941,12.91135125,13.42362312,13.98474899])
#         b_cen=np.array([12.27689266, 12.67884686, 13.16053855, 13.69871423])
#         p=np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])

#     return measures

