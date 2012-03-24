import numpy
from header import *
from error import *
from cube import *

try:
	import dicom
	_dicom_loaded = True
except:
	_dicom_loaded = False

__author__ = "Niels Bassler and Jakob Toftegaard"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"


class DosCube(Cube):
	def __init__(self,cube = None):
		super(DosCube,self).__init__(cube)
		self.type = "DOS"
	def read_dicom(self,dcm):
		if self.header_set is False:
			self.read_dicom_header(dcm)
		self.cube = []
		for i in range(len(dcm["rtdose"].pixel_array)):
			self.cube.append(dcm["images"][i].pixel_array)

