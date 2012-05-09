import numpy
from header import *
from error import *
from cube import *
from dicom.dataset import Dataset,FileDataset
from dicom.sequence import Sequence

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
		if not dcm.has_key("rtdose"):
			raise InputError, "Data doesn't contain dose infomation"
		if self.header_set is False:
			self.read_dicom_header(dcm)
		self.cube = []
		for i in range(len(dcm["rtdose"].pixel_array)):
			self.cube.append(dcm["rtdose"].pixel_array[i])
	def create_dicom_plan(self):
		meta = Dataset()
		meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
		meta.MediaStorageSOPInstanceUID = "1.2.3"
		meta.ImplementationClassUID = "1.2.3.4"
		ds = FileDataset("file", {}, file_meta=meta, preamble="\0"*128)
		ds.PatientsName = self.patient_name
		ds.PatientID = "123456"
		ds.PatientsSex = '0'
		ds.PatientsBirthDate = '19010101'
		ds.SpecificCharacterSet = 'ISO_IR 100'
		ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
		ds.StudyInstanceUID = '1.2.3'
		ds.SOPInstanceUID = '1.2.3'

		ds.Modality = "RTPLAN"
		ds.SeriesDescription = 'RT Plan'
		ds.SeriesInstanceUID = '2.16.840.1.113662.2.12.0.3057.1241703565.43' #!!!!!!!!!!
		ds.RTPlanDate = '19010101'
		ds.RTPlanGeometry = ''
		ds.RTPlanLabel = 'B1'
		ds.RTPlanTime = '000000'
		structure_ref = Dataset()
		structure_ref.RefdSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
		structure_ref.RefdSOPInstanceUID = '1.2.3'
		ds.RefdStructureSets = Sequence([structure_ref])

		dose_ref = Dataset()
		dose_ref.DoseReferenceNumber = 1
		dose_ref.DoseReferenceStructureType = 'SITE'
		dose_ref.DoseReferenceType = 'TARGET'
		dose_ref.TargetPrescriptionDose = 3.0 #Stupid
		dose_ref.DoseReferenceDescription = "TUMOR"
		ds.DoseReferences = Sequence([dose_ref])
		return ds
	def create_dicom(self):
		if _dicom_loaded is False:
			raise ModuleNotLoadedError, "Dicom"
		if self.header_set is False:
			raise InputError, "Header not loaded"
		
		ds = self.create_dicom_base()
		ds.Modality = 'RTDOSE'
	 	ds.SamplesperPixel = 1
		ds.BitsAllocated = self.num_bytes*8
		ds.BitsStored = self.num_bytes*8
		ds.AccessionNumber = ''
		ds.SeriesDescription = 'RT Dose'
		ds.DoseUnits = 'GY'
		ds.DoseType = 'PHYSICAL'
		ds.DoseGridScaling = 1.0/1000
		ds.DoseSummationType = 'PLAN'
		ds.SliceThickness = ''
		ds.InstanceCreationDate = '19010101'
		ds.InstanceCreationTime = '000000'
		ds.NumberOfFrames = len(self.cube)
		ds.PixelRepresentation = 0
		ds.StudyID = '1'
		ds.SeriesNumber = 14
		ds.GridFrameOffsetVector = self.slice_pos
		#ds.SeriesInstanceUID = '1.2.4' #!!!!!!!!!!
		ds.InstanceNumber = ''
		ds.NumberofFrames = len(self.cube)
		ds.PositionReferenceIndicator = "RF"
		ds.TissueHeterogeneityCorrection = ['IMAGE','ROI_OVERRIDE']
		ds.ImagePositionPatient = ["%.3f"%(self.xoffset*self.pixel_size), "%.3f"%(self.yoffset*self.pixel_size), "%.3f"%(self.slice_pos[0])]
		ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'
		ds.SOPInstanceUID = '1.2.246.352.71.7.320687012.47206.20090603085223'
		ds.SeriesInstanceUID = '1.2.246.352.71.2.320687012.28240.20090603082420'
		
		#Bind to rtplan
		rt_set = Dataset()
		rt_set.RefdSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'
		rt_set.RefdSOPInstanceUID = '1.2.3'
		ds.ReferencedRTPlans = Sequence([rt_set])
		pixel_array = numpy.zeros((len(self.cube),ds.Rows,ds.Columns),dtype=self.pydata_type)
		pixel_array[:][:][:] = self.cube[:][:][:]
		ds.PixelData = pixel_array.tostring()
		return ds
	def write_dicom(self,path):
		dcm = self.create_dicom()
		plan = self.create_dicom_plan()
		dcm.save_as(os.path.join(path,"rtdose.dcm"))
		plan.save_as(os.path.join(path,"rtplan.dcm"))

