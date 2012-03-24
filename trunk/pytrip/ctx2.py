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
		if _dicom_loaded is False:
			raise ModuleNotLoadedError, "Dicom"
		if self.header_set is False:
			raise InputError, "Header not loaded"
		meta = Dataset()
		meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
		meta.MediaStorageSOPInstanceUID = "1.2.3"
		meta.ImplementationClassUID = "1.2.3.4"
		ds = FileDataset("", {}, file_meta=meta, preamble="\0"*128)
		ds.Modality = 'CT'
		ds.file_meta.TransferSyntaxUID = dicom.UID.ExplicitVRBigEndian
		ds.PatientsName = self.patient_name
		ds.PatientID = "123456"
		ds.is_little_endian = True
		ds.is_implicit_VR = True
		ds.Rows = self.dimx
		ds.Columns = self.dimy
		ds.SliceThickness = self.slice_distance
		ds.BitsAllocated = self.num_bytes*8
		ds.BitsStored = self.num_bytes*8
		ds.InstanceNumber = '1'
		ds.ImagePositionPatient = [self.xoffset*self.pixel_size, self.yoffset*self.pixel_size, self.zoffset*self.slice_distance]
		ds.PixelSpacing = [self.pixel_size, self.pixel_size]
		pixel_array = numpy.zeros((ds.Rows,ds.Columns),dtype=self.pydata_type)
		for i in range(self.dimx):
			for j in range(self.dimy):
				pixel_array[i][j] = self.cube[0][i][j]
		ds.PixelData = pixel_array.tostring()
		return ds
					
			

