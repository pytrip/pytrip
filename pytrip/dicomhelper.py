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
try:
    import dicom
except:
    pass

import os


def compare_dicom_ct(dcm1, dcm2):
    diff = float(dcm1.ImagePositionPatient[2]) - float(dcm2.ImagePositionPatient[2])
    if diff > 0:
        return 1
    return -1


def read_dicom_folder(path):
    if os.path.isdir(path) is False:
        raise IOError("Folder does not exist")
    data = {}
    folder = os.listdir(path)
    for item in folder:
        if os.path.splitext(item)[1] == ".dcm":
            if (dicom.__version__ >= "0.9.5"):
                dcm = dicom.read_file(os.path.join(path, item), force=True)
            else:
                dcm = dicom.read_file(os.path.join(path, item))
            if dcm.Modality == "CT":
                if not data.has_key("images"):
                    data["images"] = []
                data["images"].append(dcm)
            elif dcm.Modality == "RTSTRUCT":
                data["rtss"] = dcm
            elif dcm.Modality == "RTDOSE":
                data["rtdose"] = dcm
            elif dcm.Modality == "RTPLAN":
                data["rtplan"] = dcm
    if data.has_key("images"):
        data["images"].sort(cmp=compare_dicom_ct)
    return data
