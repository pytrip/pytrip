import numpy
import numpy as np
from error import *
from cube import *

__author__ = "Niels Bassler and Jakob Toftegaard"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"

class LETCube(Cube):
        def __init__(self,cube = None):
                super(LETCube,self).__init__(cube)
        def calculate_lvh(self,voi): 
                n_bins = 100
                bins = np.zeros(n_bins,dtype=np.int)
                data = np.zeros(self.dimz*self.dimy*self.dimx,dtype=np.float)
                n_cube = 0
                volume = 0
                for i_z in range(self.dimz):
                        for i_y in range(self.dimy):
                                intersection = voi.get_row_intersections(self.indices_to_pos([0,i_y,i_z]))
                                if intersection is None:
                                        break;
                                if len(intersection) > 0:
                                        k = 0
                                        for i_x in range(self.dimx):
                                                if self.indices_to_pos([i_x,0,0])[0] > intersection[k]:
                                                        k = k+1
                                                        if k >= (len(intersection)):
                                                                break;
                                                        if k%2 == 1:
                                                                data[n_cube] = self.cube[i_z][i_y][i_x]
                                                                n_cube = n_cube+1
                lvh_data = data[0:n_cube]
                volume = self.pixel_size**2*self.slice_distance*n_cube
                max = np.amax(lvh_data)
                max = max + 1e-3
                for point in lvh_data:
                        i = int(point/max*n_bins)
                        bins[i] = bins[i]+1
                for i in range(n_bins-2,-1,-1):
                        bins[i] = bins[i] + bins[i+1]
                lvh = np.zeros((2,n_bins),dtype=numpy.float)

                for i in range(n_bins):
                        lvh[1][i] = float(bins[i])/float(bins[0])
                        lvh[0][i] = max/n_bins*(i+1)
                return lvh

                    
