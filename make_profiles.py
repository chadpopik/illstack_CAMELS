"""
This file handle the actual stacking, by taking in profile inputs, finding out what needs to be fetched and formatting it for illstack, and then spitting out profiles.
"""

#!/usr/bin/env python

# This contains all regular imports like numpy, pd, h5py, etc.
import sys
sys.path.append('/home/jovyan/home')
from Basics import *

sys.path.insert(0,'/home/jovyan/home/illstack_CAMELS/')
import illstack as istk

from open_sims import getICs, getgasparticles, gethalos

import stacktest



# Calculate particle values needed for whatever profiles you're computing
def particle_values(suite, box, subset, sim, snap, profs):
    # Determine which particle quantities are needed for each profile
    field_list = ['Coordinates', 'Masses']  # Always need pos and mass
    if {'gaspth', 'gastemp_uw', 'gastemp_gmw', 'gastemp_emm'} & set(profs):  # Temp/pres needs internal energy
        field_list.append('InternalEnergy')
    if {'gastemp_uw', 'gastemp_gmw', 'gastemp_emm'} & set(profs):  # Temp needs electron abundance 
        field_list.append('ElectronAbundance')
    if {'metals_uw', 'metals_gmw', 'metals_emm'} & set(profs):  # Metals needs metallicity
        if suite=='SIMBA':  # Different name in SIMBA
            field_list.append('Metallicity')
        else: 
            field_list.append('GFM_Metallicity')

    # Get gas particles from the sim, as well as ICs to get Xh
    gas_particles = getgasparticles(suite, box, subset, sim, snap, field_list)
    Xh = getICs(suite, box, subset, sim, snap)['Xh']
    
    coords, vals, volweight, weights = gas_particles['Coordinates'], [], [], []
    for p in profs:
        if p in ['gasdens', 'gasmass']:  # [1e10 Msol/h]
            vals.append(gas_particles['Masses'])
        elif p in ['gaspth']:  # [1e10 Msol/h *(km/s)**2]
            vals.append(gas_particles['Masses']*gas_particles['InternalEnergy']*(5./3.-1.))
        elif p in ['gastemp_uw', 'gastemp_gmw', 'gastemp_emm']:  # 1e10 K (km/cm)^2
            mu=(4.*c.m_p.cgs.value/(1.+3.*Xh+4.*Xh*gas_particles['ElectronAbundance']))  # CGS
            vals.append(gas_particles['InternalEnergy']*mu*(5./3.-1.)/c.k_B.cgs.value)
        elif p in ['metals_uw', 'metals_gmw', 'metals_emm']:  # [unitless ratio]
            if suite=='SIMBA':  # total metallicity, the rest are in order: He,C,N,O,Ne,Mg,Si,S,Ca,Fe
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
            weights.append(1.0+0*gas_particles['Masses'])  # Set weights=1 with right dimensions

    del gas_particles  # Posibly futile attempt to save memory

    return coords, np.array(vals), np.array(volweight), np.array(weights)



def stack(suite, box, subset, sim, snap, profs, mhlims, mass_kind, lims, scaled_radius, bins, n_jobs):
    print(f"Retrieving values for from sims for {profs} and halos")
    ppos, vals, volweight, weights = particle_values(suite, box, subset, sim, snap, profs)  # get particles values
    
    halos = gethalos(suite, box, subset, sim, snap)  # Get halo properties
    haloprops = {prop: halos[prop] for prop in halos.keys() if prop!='count'}  # Don't need count, ruins format

    ICs = getICs(suite, box, subset, sim, snap)  # Get parameters to do conversions
    h, z = ICs['HubbleParam'], ICs['Redshift']

    # Apply mass cut
    mhmin, mhmax = [mh/1e10*h for mh in mhlims]  # Convert from Msun to 1e10 Msun/h
    if mass_kind =='stellar': halosms = haloprops['GroupMassType'][:,4]  # stellar mass
    elif mass_kind =='halo': halosms = haloprops['Group_M_Crit200']  # halo mass
    halo_mask = (halosms>=mhmin) & (halosms<=mhmax)  # Mask for halo properties
    haloprops = {key: val[halo_mask] for key, val in haloprops.items()}
    
    lims = [lim*1e3*h*(1+z) for lim in lims]  # Convert from Mpc to ckpc/h
    lims[1] = np.min([ICs['BoxSize']/2, lims[1]])  # Limit outermost bin edge to half the boxsize

    print(f"Calculating Profiles for {haloprops['Group_R_Crit200'].size} Halos out to {lims[1]:.2f} ckpc/h")



    if scaled_radius is None: search_radius_in, scaled_radius_in = None, False
    else: search_radius_in, scaled_radius_in = scaled_radius, True

    profiles, counts, r_centers = istk.cyprof.stackonhalos(ppos, vals, weights, volweight, haloprops['GroupPos'], haloprops['Group_R_Crit200'], ICs['BoxSize'], scaled_radius_in, search_radius_in, lims, bins)
    
    # profiles, counts, r_centers = stacktest.stacknew(ppos, vals, weights, volweight, haloprops['GroupPos'], haloprops['Group_R_Crit200'], ICs['BoxSize'], scaled_radius_in, search_radius_in, lims, bins, n_jobs)

    # Reshape the output
    haloprofs = {'r': r_centers, 'npart': counts}
    for i in range(len(profs)):
        haloprofs[profs[i]] = profiles[i]
    
    return haloprofs, haloprops, ICs
