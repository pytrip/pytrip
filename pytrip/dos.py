"""
    This file is part of PyTRiP.

    libdedx is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libdedx is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libdedx.  If not, see <http://www.gnu.org/licenses/>
"""
import numpy
from pytrip.header import *
from pytrip.error import *
from pytrip.cube import *

import pytrip as plib
import time
import pytrip.res.point
import gc

try:
    import dicom
    from dicom.dataset import Dataset, FileDataset
    from dicom.sequence import Sequence

    _dicom_loaded = True
except:
    _dicom_loaded = False

__author__ = "Niels Bassler and Jakob Toftegaard"
__version__ = "1.0"
__email__ = "bassler@phys.au.dk"


def calculate_dose_cube(field, density_cube, isocenter, pre_dose, pathcube=None, factor=1.0):
    cube_size = [density_cube.pixel_size, density_cube.pixel_size, density_cube.slice_distance]
    basis = field.get_cube_basis()

    if pathcube is None:
        pathcube = plib.rhocube_to_water(numpy.array(density_cube.cube), numpy.array(basis[0]), numpy.array(cube_size))
        pathcube += field.bolus

    dist = plib.calculate_dist(pathcube, numpy.array(cube_size), isocenter, numpy.array(basis))
    field_size = field.field_size

    dist = numpy.reshape(dist, (density_cube.dimx * density_cube.dimy * density_cube.dimz, 3))
    raster_matrix = np.array(field.get_merged_raster_points())

    ddd = field.get_ddd_list()
    dose = plib.calculate_dose(dist, numpy.array(raster_matrix), numpy.array(ddd));

    dose = numpy.reshape(dose, (density_cube.dimz, density_cube.dimy, density_cube.dimx)) * 1.602189 * 10 ** -8
    dos = DosCube(density_cube)
    dos.cube = numpy.array(dose / pre_dose * 1000 * factor, dtype=numpy.int16)
    gc.collect()
    return (dos, pathcube)


class DosCube(Cube):
    def __init__(self, cube=None):
        super(DosCube, self).__init__(cube)
        self.type = "DOS"
        self.target_dose = 0.0

    def read_dicom(self, dcm):
        if not dcm.has_key("rtdose"):
            raise InputError("Data doesn't contain dose infomation")
        if self.header_set is False:
            self.read_dicom_header(dcm)
        self.cube = numpy.zeros((self.dimz, self.dimy, self.dimx))
        for i in range(len(dcm["rtdose"].pixel_array)):
            self.cube[i][:][:] = dcm["rtdose"].pixel_array[i]

    def calculate_dvh(self, voi):
        pos = 0
        size = numpy.array([self.pixel_size, self.pixel_size, self.slice_distance])
        dv = numpy.zeros(1500)
        valid = False
        for i in range(self.dimz):
            pos += self.slice_distance
            slice = voi.get_slice_at_pos(pos)
            if slice is not None:
                valid = True
                dv += plib.calculate_dvh_slice(self.cube[i], numpy.array(slice.contour[0].contour), size)
        if valid:
            cubes = sum(dv)
            dvh = numpy.cumsum(dv[::-1])[::-1] / cubes

            min_dose = numpy.where(dvh >= 0.98)[0][-1]
            max_dose = numpy.where(dvh <= 0.02)[0][0]
            area = cubes * size[0] * size[1] * size[2] / 1000
            mean = numpy.dot(dv, range(0, 1500)) / cubes
            return (dvh, min_dose, max_dose, mean, area)
        return None

    # ~ print time.time()-starttime


    def create_dicom_plan(self):
        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        meta.MediaStorageSOPInstanceUID = "1.2.3"
        meta.ImplementationClassUID = "1.2.3.4"
        ds = FileDataset("file", {}, file_meta=meta, preamble="\0" * 128)
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
        ds.SeriesInstanceUID = '2.16.840.1.113662.2.12.0.3057.1241703565.43'  # !!!!!!!!!!
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
        dose_ref.TargetPrescriptionDose = self.target_dose
        dose_ref.DoseReferenceDescription = "TUMOR"
        ds.DoseReferences = Sequence([dose_ref])
        return ds

    def create_dicom(self):
        if _dicom_loaded is False:
            raise ModuleNotLoadedError("Dicom")
        if self.header_set is False:
            raise InputError("Header not loaded")

        ds = self.create_dicom_base()
        ds.Modality = 'RTDOSE'
        ds.SamplesperPixel = 1
        ds.BitsAllocated = self.num_bytes * 8
        ds.BitsStored = self.num_bytes * 8
        ds.AccessionNumber = ''
        ds.SeriesDescription = 'RT Dose'
        ds.DoseUnits = 'GY'
        ds.DoseType = 'PHYSICAL'
        ds.DoseGridScaling = self.target_dose / 10 ** 5
        ds.DoseSummationType = 'PLAN'
        ds.SliceThickness = ''
        ds.InstanceCreationDate = '19010101'
        ds.InstanceCreationTime = '000000'
        ds.NumberOfFrames = len(self.cube)
        ds.PixelRepresentation = 0
        ds.StudyID = '1'
        ds.SeriesNumber = 14
        ds.GridFrameOffsetVector = [x * self.slice_distance for x in range(self.dimz)]
        # ds.SeriesInstanceUID = '1.2.4' #!!!!!!!!!!
        ds.InstanceNumber = ''
        ds.NumberofFrames = len(self.cube)
        ds.PositionReferenceIndicator = "RF"
        ds.TissueHeterogeneityCorrection = ['IMAGE', 'ROI_OVERRIDE']
        ds.ImagePositionPatient = ["%.3f" % (self.xoffset * self.pixel_size), "%.3f" % (self.yoffset * self.pixel_size),
                                   "%.3f" % (self.slice_pos[0])]
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.2'
        ds.SOPInstanceUID = '1.2.246.352.71.7.320687012.47206.20090603085223'
        ds.SeriesInstanceUID = '1.2.246.352.71.2.320687012.28240.20090603082420'

        # Bind to rtplan
        rt_set = Dataset()
        rt_set.RefdSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5'
        rt_set.RefdSOPInstanceUID = '1.2.3'
        ds.ReferencedRTPlans = Sequence([rt_set])
        pixel_array = numpy.zeros((len(self.cube), ds.Rows, ds.Columns), dtype=self.pydata_type)
        pixel_array[:][:][:] = self.cube[:][:][:]
        ds.PixelData = pixel_array.tostring()
        return ds

    def write(self, path):
        f_split = os.path.splitext(path)
        header_file = f_split[0] + ".hed"
        dos_file = f_split[0] + ".dos"
        self.write_trip_header(header_file)
        self.write_trip_data(dos_file)

    def write_dicom(self, path):
        dcm = self.create_dicom()
        plan = self.create_dicom_plan()
        dcm.save_as(os.path.join(path, "rtdose.dcm"))
        plan.save_as(os.path.join(path, "rtplan.dcm"))
