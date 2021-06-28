basepath = '/home/jovyan/Simulations/IllustrisTNG/1P_22/'
z = 0.53726 
serial      = True
search_radius = 12.5
lims        = [10e-2*(1.+z),10e4*(1.+z)]  #[0.01,10] scaled
bins        = 25
mass_low    = 10**11.0  #actually 10^11/h
mass_high   = 10**15.0  #actually 10^15/h 
scaled_radius = False #True == scaled, False == unscaled
mass_kind   = 'halo' #options='stellar','halo' 
save_direct = '/home/jovyan/home/illstack/CAMELS_example/Batch_NPZ_files_with_ID/' 
