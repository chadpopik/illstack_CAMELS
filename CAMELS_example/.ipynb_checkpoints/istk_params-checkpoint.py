basepath = '/home/jovyan/OLD1P/Sims/IllustrisTNG/1P/1P_15/'
z = 0.53726 
serial      = True
search_radius = 15.
lims        = [1e-2*1e3, 12.5*1e3]  #[0.01,10] scaled
bins        = 50
mass_low    = 10**11.0  #actually 10^11/h
mass_high   = 10**15.0  #actually 10^15/h 
scaled_radius = False #True == scaled, False == unscaled
mass_kind   = 'halo' #options='stellar','halo' 
save_direct = '/home/jovyan/home/illstack_CAMELS/CAMELS_example/Batch_hdf5_files/IllustrisTNG/1P/'
h = 0.6711