import sys
sys.path.insert(0,'/home/jovyan/home/')
sys.path.append('/home/jovyan/home/CAMELS_stuff/illustris_python')
import illustris_python as il
import numpy as np
from . import params





def getparticles(snapshot_number,partType,field_list):

    basePath=params.basepath
    particles = il.snapshot.loadSubset(basePath,snapshot_number,partType,fields=field_list)
    return particles

def gethalos(snapshot_number,field_list):
    
    basePath=params.basepath
    halos=il.groupcat.loadHalos(basePath,snapshot_number,fields=field_list)
    return halos

def getsubhalos(snapshot_number,field_list):
    basePath=params.basepath
    subhalos=il.groupcat.loadSubhalos(basePath, snapshot_number,fields=field_list)
    return subhalos


