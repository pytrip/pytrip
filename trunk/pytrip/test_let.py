from let import * 
from vdx2 import *
from numpy import *
import matplotlib.pyplot as plt
l = LETCube()
l.read_trip_data_file("/home/jato/Projects/TRiP/2012_TEST/1/test.dosemlet.dos")

v = VdxCube("")
v.import_vdx("/home/jato/Projects/TRiP/2012_TEST/1/test.vdx")
tumor = v.get_voi_by_name("Tumor Bed")
lvh = l.calculate_lvh(tumor)



plt.plot(lvh[0],lvh[1])
plt.show()
