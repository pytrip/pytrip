from numpy import *
from ctx2 import *
from cube import *
from scipy import interpolate
from math import *
import time

class DensityProjections:
    def __init__(self,cube):
        self.cube = cube
    #voi is tumortarget and type is Voi, gantry and couch are in degree
    #stepsize is relative to pixelsize 1 is a step of 1 pixel
    def calculate_projection(self,voi,gantry,couch,stepsize=1.0):

        #Convert angles from degrees to radians
        min_structure = 5
        self.gantry = gantry*pi/180
        self.couch = couch*pi/180
        #calculate surface normal
        self.step_vec = stepsize*self.cube.pixel_size*numpy.array([cos(self.gantry)*sin(self.couch),sin(self.gantry),cos(self.couch)*cos(self.gantry)])
        self.step_length = stepsize*self.cube.pixel_size
        center = voi.calculate_center()
        (b,c) = self.calculate_plane_vectors(self.gantry,self.couch)
        min_window,max_window = voi.get_max_min()
        size = numpy.array(max_window)-numpy.array(min_window)

        window_size = numpy.array([((cos(self.gantry)*size[1])**2+(sin(self.gantry)*size[2])**2)**0.5,((cos(self.couch)*size[0])**2+(sin(self.couch)*size[2])**2)**0.5])*1.50

        dimension = window_size/self.cube.pixel_size
        dimension = numpy.int16(dimension)
        start = center-self.cube.pixel_size*0.5*numpy.array(dimension[0]*b+dimension[1]*c)
        #speed up python
        cube = self.cube
        step_length = self.step_length
        step_vec = self.step_vec
        i = 0
        temp = start
        temp = center.copy()
        length = 0.0
        j = None
        #Calculate
        try:
            while True:
                value = cube.get_value_at_pos(temp)
                temp += step_vec
                if sum(temp < 0) > 0:
                    break
                i += 1
                if value < 0.1 and j == None:
                    j = i
                    length = 0.0
                if value > 0.1:
                    length += step_length
                    if length > 4.0:
                        j = None
        except:
            a = 0
        if j == None:
            j = i
        j = j*1.10
        if j > i:
            j=i-30
        j = int(j)
        #create step vector
        steps = [step_vec*i for i in range(j)]
        #ini points array
        points = numpy.zeros((dimension[1],dimension[0],j,3),dtype=float)
        points_int = numpy.zeros((dimension[1],dimension[0],j,3),dtype=int)

        #calculate x vector
        x_vector = numpy.array([[i*b]*j for i in range(dimension[0])])
        #calculate position matrix
        for y in range(dimension[1]):
            line = (x_vector+y*c)*cube.pixel_size
            line = line+start
            line = line[:] + steps
            points[y,:,:] = line
        #convert position matrix to indices
        points_int[:,:,:,0] = numpy.array(points[:,:,:,0]/cube.pixel_size,dtype=int)
        points_int[:,:,:,1] = numpy.array(points[:,:,:,1]/cube.pixel_size,dtype=int)
        points_int[:,:,:,2] = numpy.array(points[:,:,:,2]/cube.slice_distance,dtype=int)
        
        points_int[:,:,:,0] = (points_int[:,:,:,0] < self.cube.dimx)*points_int[:,:,:,0]
        points_int[:,:,:,1] = (points_int[:,:,:,1] < self.cube.dimy)*points_int[:,:,:,1]
        points_int[:,:,:,2] = (points_int[:,:,:,2] < self.cube.dimz)*points_int[:,:,:,2]
        
        #lookup densitys
        densitys = cube.cube[points_int[:,:,:,2],points_int[:,:,:,1],points_int[:,:,:,0]]
        #sum density
        data = numpy.sum(densitys,2)*self.step_length
        return numpy.transpose(data),start,[b,c]

    def calculate_plane_vectors(self,gantry,couch):
        a = numpy.array([cos(self.gantry)*sin(self.couch),sin(self.gantry),cos(self.couch)*cos(self.gantry)])
        b = numpy.array([cos(gantry+pi/2)*sin(couch),sin(gantry+pi/2),cos(couch)*cos(gantry+pi/2)])
        c = numpy.cross(a,b)
        return (b,c)


class DensityCube(Cube):
    def __init__(self,ctxcube):
        self.ctxcube = ctxcube
        super(DensityCube,self).__init__(ctxcube)
        self.type = "Density"
        self.directory = os.path.dirname(os.path.abspath( __file__ ))
        self.hlut_file = self.directory + "/data/hlut_den.dat"
        self.import_hlut()
        self.calculate_cube()

    def calculate_cube(self):
        ctxdata = self.ctxcube.cube
        ctxdata = ctxdata.reshape(self.dimx*self.dimy*self.dimz)
        cube = interpolate.splev(ctxdata,self.hlut_data)
        cube = cube.astype(numpy.float32)
        self.cube = np.reshape(cube,(self.dimz,self.dimy,self.dimx))

    def import_hlut(self):
        fp = open(self.hlut_file,"r")
        lines = fp.read();
        fp.close()
        lines = lines.split('\n')
        x_data = []
        y_data = []
        for line in lines:
            a = line.split()
            if len(a):
                x_data.append(a[0])
                y_data.append(a[3])
        self.hlut_data = interpolate.splrep(x_data,y_data,s=0)
