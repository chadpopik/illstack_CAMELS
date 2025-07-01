import sys
sys.path.append('/home/jovyan/home/')
from Basics import *

savepath='/home/jovyan/home/emu_CAMELS/emulator_profiles'
filepath='/home/jovyan/home/Profiles/'

#-----------------------------------input section
suite="IllustrisTNG"
subset="1P"




for sim in simulations:
    for snap in snapshots:
        r, valprofiles, haloprops, simparams = extract(suite, subset, sim, snap)

        file_hdf5=h5py.File(f"{save_direct}/{suite}_{subset}_{sim}_{snap}.hdf5",'w',track_order=True)
        for prof in profs.keys():
            file_hdf5.create_dataset(f'{prof}/Mean', data=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), 0, profs[prof]), dtype='float64')
            file_hdf5.create_dataset(f'{prof}/STD', data=np.apply_along_axis(lambda v: np.std(v[np.nonzero(v)]), 0, profs[prof]), dtype='float64')
            file_hdf5.create_dataset(f'{prof}/P50', data=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],50), 0, profs[prof]), dtype='float64')
            file_hdf5.create_dataset(f'{prof}/P16', data=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],16), 0, profs[prof]), dtype='float64')
            file_hdf5.create_dataset(f'{prof}/P80', data=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],84), 0, profs[prof]), dtype='float64')

            if weights are given:
                ws = weights[np.searchsorted(bin_edges, np.log10(weightby))-1]
                measures['weighted'] = np.apply_along_axis(lambda v: np.average(v[np.nonzero(v)], weights=ws[np.nonzero(v)]), 0, prof)

        b_edges=np.array([12.11179316,12.46636941,12.91135125,13.42362312,13.98474899])
        b_cen=np.array([12.27689266, 12.67884686, 13.16053855, 13.69871423])
        p=np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])

    return measures





def cut(r, profs, haloprops, mh, mh_low=0, mh_high=np.inf, inner_cut=0, outer_cut=np.inf):  # Cut by mass and radius
    mcut = (mh > mh_low) & (mh < mh_high)
    rcut = (r <= outer_cut) & (r >= inner_cut) & (np.sum(profs['nparticles'][mcut], axis=0)!=0)
    cutprofs = {prof: profs[prof][mcut, :][:, rcut] for prof in profs.keys()}
    cutprops = {prop: haloprops[prop][mcut] for prop in haloprops.keys()}
    return r[rcut], cutprofs, cutprops


def extract(suite, subset, sim, snap):  # Load halo profiles and propertie sfrom illstack outputs and assign proper units
    if subset=='LH':  # Subfolders are clearly named for LH set, but the data files use z snap numbers in the 91 format
        subfolder = f'LH_{sim}'
        datasnap = zdf.loc[zdf['34 snapshots']==int(snap), '91 snapshots'].iloc[0]
        simparams = h5py.File(f'/home/jovyan/Data/Sims/IllustrisTNG/LH/LH_{sim}/snapshot_{datasnap:03}.hdf5', 'r')['Header'].attrs
    elif subset=='1P':  # 1P alternates between naming conventions and requires some translating
        if sim%11-5<0: 
            subfolder = f'1P_{(np.ceil(sim/10+0.1)):.0f}_n{np.abs(sim%11-5)}'
        else: 
            subfolder = f'1P_{(np.ceil(sim/11)):.0f}_{sim%11-5}'
        if sim%11-5==0:  # The fiducial values aren't in the OLD1P so have to fetch from the new ones
            datasnap = zdf.loc[zdf['34 snapshots']==int(snap), '91 snapshots'].iloc[0]
            simparams = h5py.File(f'/home/jovyan/Data/Sims/{suite}/{subset}/1P_p{(np.ceil(sim/10)):.0f}_{sim%11-5}/snapshot_{datasnap:03}.hdf5', 'r')['Header'].attrs
        else:
            simparams = h5py.File(f'/home/jovyan/OLD1P/Sims/{suite}/{subset}/{subfolder}/snap_{snap:03}.hdf5', 'r')['Header'].attrs


    profiles = h5py.File(f'/home/jovyan/Data/Profiles/{suite}/{subset}/{subfolder}/{suite}_{subset}_{sim}_{snap:03}.hdf5', 'r')

    r = profiles['r'][()] /(1+z)/h/1.e3
    valprofiles = {
        'dens' : profiles['Profiles'][0] *1e10*1e3**3*(1+z)**3*h**2,  # (Msun/h)/(kpc/h)^3 to Msun/Mpc^3
        'pres' : profiles['Profiles'][1]  *1e10*(u.km**2/u.kpc**3).to(1/u.Mpc)*(1+z)**3*h**2,  # (Msun/h)*(km/s)^2/(ckpc/h)**3 to Msun/Mpc/s^2
        'metal_gmw' : profiles['Profiles'][2],  # Ratio
        'temp_gmw' : profiles['Profiles'][3] *1.e10,  # K
        'nparticles' : profiles['n'][0]  # Number
            }

    halopropsraw = {key:profiles[key][()] for key in list(profiles.keys()) if profiles[key].shape==profiles['ID'].shape}
    haloprops = convert(halopropsraw, simparams['Redshift'], simparams['HubbleParam'])

    return r, valprofiles, haloprops, dict(simparams)


def convert(haloprops, z, h):  # General function to convert the units from sims/profiles out of comoving/littleh units, would be good to continue to generalize
    newhaloprops = {}
    for prop in haloprops.keys():
        if prop in ['Group_M_Crit200', 'GroupMassType_Stellar', 'GroupBHMass', 'GroupMass', 'GroupWindMass', 'Group_M_Crit500', 'Group_M_Mean200', 'Group_M_TopHat200']:
            newhaloprops[prop] = haloprops[prop][()] *1e10/h  # 1e10 Msol/h to Msun
        elif prop in ['Group_R_Crit200', 'Group_R_Crit500', 'Group_R_Mean200','Group_R_TopHat200']:
            newhaloprops[prop] = haloprops[prop][()] /1e3/(1+z)/h  # ckpc/h to Mpc
        elif prop in ['GroupVelx', 'GroupVely', 'GroupVelz']:
            newhaloprops[prop] = haloprops[prop][()] *(1+z)  # km/s/a to km/s
        elif prop=='GroupBHMdot':
            newhaloprops[prop] = haloprops[prop][()] *1e10/(0.978*1e9)  # 1e10 (Msol/h)/(0.978 Gyr/h) to Msun/yr
        else:
            newhaloprops[prop] = haloprops[prop][()]
    return newhaloprops