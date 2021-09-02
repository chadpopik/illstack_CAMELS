import numpy as np

home='/home/jovyan/home/illstack/CAMELS_example/Tests/SR_tests/'
suite='SIMBA'

snap=['024','025','026','027','028','029','030','031','032','033']

for s in np.arange(len(snap)):
    
    SR,max_rh=np.genfromtxt(home+'sr_test_'+suite+'_'+snap[s]+'.txt',usecols=(5,8),unpack=True)
    print(snap[s],SR)
    
    
    #print("For %s snap %s we need SR %.2f"%(suite,snap[s],max(SR)))