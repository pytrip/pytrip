from pytrip import *
from numpy import array
from enthought.mayavi import mlab
#from enthought.mayavi.api import Engine
ct = CtxCube("testfiles/CBS303000.ctx")
ds = DoseCube("testfiles/CBS303101.dos")

#fancy transparent plots
#mlab.pipeline.volume(mlab.pipeline.scalar_field(ct.cube))
#mlab.pipeline.volume(mlab.pipeline.scalar_field(ds.cube))

#solid surface
obj = mlab.contour3d(ct.cube, transparent=False, contours=2, extent=[0,256,0,256,0,256])

mlab.show()
