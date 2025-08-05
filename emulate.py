#Saving and loading the emulator
def save_emulator(filename, radius, emulator):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump((radius, emulator), f) 
    
def load_emulator(filename) :
    import pickle
    with open(filename, 'rb') as f:
        radius, emulator = pickle.load(f) 
    return radius, emulator



emulator=ostrich.emulate.PcaEmulator.create_from_data(
    samples,  # set of sampled parameters, in a cartesian products so you scan over every combination of parameters, with dim=[nprofs, nparams]
    y,  # set of y values, or logy values, with dim=[nr, nprofs]
    ostrich.interpolate.RbfInterpolator,  # choice of interpolator
    interpolator_kwargs={'function':func_str},  # choice of inerpolator, like 'linear'
    num_components=12  # number of basis vectors to use?
)  
return samples,x,y,emulator