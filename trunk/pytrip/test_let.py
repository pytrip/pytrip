from let import * 
from numpy import *
l = LETCube()
l.read_trip_data_file("/home/jato/Desktop/2011_NCPRT/foo_ptv000.dosemlet.dos")
print amax(l.cube)
