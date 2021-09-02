import numpy as np
import readgadget

tng='/home/jovyan/Simulations/IllustrisTNG/1P_0/ICs/ics'
smb='/home/jovyan/Simulations/SIMBA/1P_0/ICs/ics'

header_tng=readgadget.header(tng)
header_smb=readgadget.header(smb)

h_tng=header_tng.hubble
h_smb=header_smb.hubble

print(h_tng)
print(h_smb)

