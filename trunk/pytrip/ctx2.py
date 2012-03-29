import numpy
from header import *
from error import *
from cube import *
from dicom.dataset import Dataset, FileDataset

try:
	import dicom
	_dicom_loaded = True
except:
	_dicom_loaded = False

__author__ = "Niels Bassler and Jakob Toftegaard"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"


class CtxCube(Cube):
	def __init__(self,cube = None):
		super(CtxCube,self).__init__(cube)
		self.type = "CTX"
	def read_dicom(self,dcm):
		if self.header_set is False:
			self.read_dicom_header(dcm)
		self.cube = []
		for i in range(len(dcm["images"])):
			self.cube.append(dcm["images"][i].pixel_array)
	def create_dicom(self):
		data = []
		if _dicom_loaded is False:
			raise ModuleNotLoadedError, "Dicom"
		if self.header_set is False:
			raise InputError, "Header not loaded"
		
		for i in range(len(self.cube)):
			ds = self.create_dicom_base()
			ds.Modality = 'CT'
	 		ds.SamplesPerPixel = 1
			ds.BitsAllocated = self.num_bytes*8
			ds.BitsStored = self.num_bytes*8
			ds.PatientPosition = 'HFS'
			ds.RescaleIntercept = 0
			ds.RescaleSlope = 1
			ds.PixelRepresentation = 1
			ds.ImagePositionPatient = ["%.3f"%(self.xoffset*self.pixel_size), "%.3f"%(self.yoffset*self.pixel_size), "%.3f"%(self.slice_pos[i])]
			ds.SliceLocation = str(self.slice_pos[i])
			ds.InstanceNumber = str(i+1)
			pixel_array = numpy.zeros((ds.Rows,ds.Columns),dtype=self.pydata_type)
			pixel_array[:][:] = self.cube[0][:][:]
			ds.PixelData = pixel_array.tostring()
			data.append(ds)
		return data
	def write_dicom(self,path):
		dcm_list = self.create_dicom()
		for i in range(len(dcm_list)):
			dcm_list[i].save_as(os.path.join(path,"ct.%d.dcm"%(dcm_list[i].InstanceNumber-1)))
