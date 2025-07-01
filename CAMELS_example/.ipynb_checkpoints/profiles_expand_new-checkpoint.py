#!/usr/bin/env python

# This contains all regular imports like numpy, pd, h5py, etc.
import sys
sys.path.append('/home/jovyan/home/')
from Basics import *

sys.path.insert(0,'/home/jovyan/home/illstack_CAMELS/')
import illstack as istk

sys.path.append('/home/jovyan/home/illustris_python')
import illustris_python as il

sys.path.append('/home/')
from jovyan.illustris_python import illustris_python as ilnew



# Calculate particle values needed for whatever profiles you're computing
def particle_values(basepath, suite, subset, sim, snap, profs):
    field_list = ['Coordinates', 'Masses']  # Always need position and mass
    if {'gaspth', 'gastemp_uw', 'gastemp_gmw', 'gastemp_emm'} & set(profs):  # If calculating temp or pres, need internal energy
        field_list.append('InternalEnergy')
    if {'gastemp_uw', 'gastemp_gmw', 'gastemp_emm'} & set(profs):  # If calculating temperature, need electron abundance 
        field_list.append('ElectronAbundance')
    if {'metals_uw', 'metals_gmw', 'metals_emm'} & set(profs):  # If calculating metals, need metallicity
        if suite=='SIMBA': 
            field_list.append('Metallicity')  # Has a different name in SIMBA
        else: 
            field_list.append('GFM_Metallicity')


    print(f"Fetching particle quantities for {field_list}")  # NOTE: These sections contain path details which may need to be altered
    if suite=='TNG_L50':  # Slightly different format and io functions due to chunking
        simpath = f"{basepath}/Sims/{suite}_{subset}/{subset}_{sim}/"
        gas_particles = ilnew.snapshot.loadSubset(simpath, snap, 'gas', field_list)
        simparams = dict(h5py.File(f"{simpath}/snapdir_{snap:03}/snap_{snap:03}.0.hdf5", 'r')['Header'].attrs)
        astroparamsdf = pd.read_csv(f"/home/jovyan/PUBLIC_RELEASE/Parameters/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L50n512_{subset}.txt", sep='\s+')  # Bit of a lazy fix
        astroparams = astroparamsdf.loc[astroparamsdf['#Name']==f"{subset}_{sim}"].to_dict('records')[0]
    else:
        simpath = f"{basepath}/Sims/{suite}/{subset}/{subset}_{sim}/"
        gas_particles=il.snapshot.loadSubset(simpath, snap,'gas', field_list)
        simparams = dict(h5py.File(f"{simpath}/snapshot_{snap:03}.hdf5", 'r')['Header'].attrs)
        astroparamsdf = pd.read_csv(f"/home/jovyan/PUBLIC_RELEASE/Parameters/{suite}/CosmoAstroSeed_IllustrisTNG_L25n256_{subset}.txt", sep='\s+')
        astroparams = astroparamsdf.loc[astroparamsdf['#Name']==f"{subset}_{sim}"].to_dict('records')[0]

    for line in open(f"{simpath}/ICs/CAMB.params", 'r'):  # Fetching Xh directly from the ICs, instead of assuming 0.76
        if line.strip().startswith("YHe"): 
            simparams['Xh'] = 1-np.float64(line.strip().split('=', 1)[-1])

    print(f"Computing values for {profs}")
    coords, vals, volweight, weights = gas_particles['Coordinates'], [], [], []
    for p in profs:
        if p in ['gasdens', 'gasmass']:  # units 1e10 Msol/h
            vals.append(gas_particles['Masses'])
        elif p in ['gaspth']:  # units 1e10Msol/h*(km/s)**2
            vals.append(gas_particles['Masses']*gas_particles['InternalEnergy']*(5./3.-1.))
        elif p in ['gastemp_uw', 'gastemp_gmw', 'gastemp_emm']:  # 1e10*K*(km/cm)^2
            mu=(4.*c.m_p.cgs.value/(1.+3.*Xh+4.*Xh*gas_particles['ElectronAbundance']))  # CGS
            vals.append(gas_particles['InternalEnergy']*mu*(5./3.-1.)/c.k_B.cgs.value)
        elif p in ['metals_uw', 'metals_gmw', 'metals_emm']:  # metallicity ratio
            if suite=='SIMBA':  # total metalallicity, the rest are in order: He,C,N,O,Ne,Mg,Si,S,Ca,Fe
                vals.append(gas_particles['Metallicity'][:,0])
            else:
                vals.append(gas_particles['GFM_Metallicity'])
        else:  # If we get an undefined profile
            print("Please enter an appropriate option for the profile")

        if p in ['gasdens', 'gaspth']:  # Pressure and density must be volume weighted
            volweight.append(True)
        elif p in ['metals_uw', 'metals_gmw', 'gasmass', 'gastemp_uw', 'gastemp_gmw', 'metals_emm', 'gastemp_emm']:
            volweight.append(False)

        if p in ['metals_gmw','gastemp_gmw']:  # GNW Metallicity and GNW Temp must be mass weighted
            weights.append(gas_particles['Masses'])
        elif p in ['gasdens', 'gaspth', 'metals_uw', 'gasmass','gastemp_uw', 'metals_emm', 'gastemp_emm']:
            weights.append(1.0+0*gas_particles['Masses'])

    del gas_particles  # Posibly futile attempt to save memory

    return coords, np.array(vals), np.array(volweight), np.array(weights), simparams | astroparams



# Calculate halo profiles
def stack(basepath, suite, subset, sim, snap, profs, lims, bins, mhlims, mass_kind):
    # Get particle positions, values, and weights from previous function
    ppos, vals, volweight, weights, simparams = particle_values(basepath, suite, subset, sim, snap, profs)

    # NOTE: These sections contain path details which may need to be altered
    print("Fetching halo properties")
    if suite=="TNG_L50":
        halopath = f"{basepath}/FOF_Subfind/{suite}_{subset}/{subset}_{sim}/"
        halos = ilnew.groupcat.loadHalos(halopath, snap)
        haloparams = dict(h5py.File(f"{halopath}/groups_{snap:03}/fof_subhalo_tab_{snap:03}.0.hdf5", 'r')['Parameters'].attrs)
    else:
        halopath = f"{basepath}/Sims/{suite}/{subset}/{subset}_{sim}/"
        halos = il.groupcat.loadHalos(halopath, snap)
        haloparams = dict(h5py.File(f"{halopath}/groups_{snap:03}.hdf5", 'r')['Parameters'].attrs)
    haloprops = {prop: halos[prop] for prop in halos.keys() if prop!='count'}  # Don't need count

    # Convert some parameteres and set others
    mhmin, mhmax = [mh/1e10*simparams['HubbleParam'] for mh in mhlims]  # Convert to 1e10Msun/h
    lims = [lim*1e3*simparams['HubbleParam']*(1+simparams['Redshift']) for lim in lims]  # Convert to ckpc/h
    lims[1] = np.min([simparams['BoxSize']/2, lims[1]])  # Don't allow for upper limit greater than half the box size
    search_radius = lims[1]/np.max(haloprops['Group_R_Crit200'])  # Set search radius for particles equal to the outer lim to avoid empty bins
    
    scaled_radius = False  # Whether to use scaled radius x=r/r200c
    ntile = 3  # controls tiling -- optimal when each tile has a few halos

    print(f"Calculating Profiles for {haloprops['Group_M_Crit200'].size} Halos out to {lims[1]:.2f} ckpc/h")
    pcen, pval, pnum, nhalos, phaloprops = istk.cyprof.stackonhalos(posp = ppos, vals = vals, volweight = volweight, weights = weights, box = simparams['BoxSize'], posh = haloprops['GroupPos'], mh = haloprops['Group_M_Crit200'], rh = haloprops['Group_R_Crit200'], mstar = haloprops['GroupMassType'][:,4], haloprops = haloprops, mhmin = mhmin,  mhmax = mhmax, mass_kind = mass_kind, scaled_radius = scaled_radius, search_radius = search_radius, lims = lims, bins = bins, ntile = ntile, rank = 0)

    # Reshape the profiles
    haloprofs = {'r': np.reshape(pcen,  (nhalos,bins)), 'npart': np.reshape(pnum, (pval.shape[0], nhalos, bins))[0]}
    for i in range(len(profs)):
        haloprofs[profs[i]] = np.reshape(pval[i], (nhalos, bins))
    
    return haloprofs, phaloprops, simparams | haloparams



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




# Extract profiles from old ones that Emily ran
def extract_old(basepath, suite, subset, sim, snap):
    # The profiles and old 1P sims use a snap out of 34 instead of the 91 ordering, so have to convert
    zdf = pd.read_csv(f'/home/jovyan/home/illstack_CAMELS/zs.csv')
    oldsnap = int(zdf.loc[zdf['91 snapshots']==int(snap), '34 snapshots'].iloc[0])

    # Get out simparams for conversion purposes
    if subset=='LH':
        subfolder = f'LH_{sim}'
        simparams = h5py.File(f'{basepath}/Sims/IllustrisTNG/LH/{subfolder}/snapshot_{snap:03}.hdf5', 'r')['Header'].attrs
    elif subset=='1P':  # 1P alternates between naming conventions and requires some translating
        if sim%11-5<0: 
            subfolder = f'1P_{(np.ceil(sim/10+0.1)):.0f}_n{np.abs(sim%11-5)}'
        else: 
            subfolder = f'1P_{(np.ceil(sim/11)):.0f}_{sim%11-5}'
        if sim%11-5==0:  # The fiducial values aren't in the OLD1P so have to fetch from the new ones
            simparams = h5py.File(f'{basepath}/Sims/{suite}/{subset}/1P_p{(np.ceil(sim/10)):.0f}_{sim%11-5}/snapshot_{snap:03}.hdf5', 'r')['Header'].attrs
        else:
            simparams = h5py.File(f'/home/jovyan/OLD1P/Sims/{suite}/{subset}/{subfolder}/snap_{oldsnap:03}.hdf5', 'r')['Header'].attrs

    # Load profiles and properties into dictionaries
    profiles = h5py.File(f'{basepath}/Profiles/{suite}/{subset}/{subfolder}/{suite}_{subset}_{sim}_{oldsnap:03}.hdf5', 'r')
    haloprops = {key:profiles[key][()] for key in list(profiles.keys()) if profiles[key].shape==profiles['ID'].shape}
    valprofiles = {'r' : profiles['r'][()]*np.ones(profiles['n'][0].shape),
                   'npart' : profiles['n'][0],
                   'dens' : profiles['Profiles'][0], 'pres' : profiles['Profiles'][1],'metal_gmw' : profiles['Profiles'][2],'temp_gmw' : profiles['Profiles'][3]}        

    return valprofiles, haloprops, dict(simparams)