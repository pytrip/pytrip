"""
    This file is part of PyTRiP.

    PyTRiP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP.  If not, see <http://www.gnu.org/licenses/>
"""
import os
import dicom


def compare_dicom_key(dcm):
    return float(dcm.ImagePositionPatient[2])


def read_dicom_folder(path):
    if not os.path.isdir(path):
        raise IOError("Folder does not exist")
    data = {}
    folder = os.listdir(path)
    for item in folder:
        if os.path.splitext(item)[1] == ".dcm":
            dcm = dicom.read_file(os.path.join(path, item), force=True)
            # TODO figureout what was it about (see below)
            # if dicom.__version__ >= "0.9.5":
            # dcm = dicom.read_file(os.path.join(path, item), force=True)
            # else:
            #     dcm = dicom.read_file(os.path.join(path, item))
            if dcm.Modality == "CT":
                if "images" not in data:
                    data["images"] = []
                data["images"].append(dcm)
            elif dcm.Modality == "RTSTRUCT":
                data["rtss"] = dcm
            elif dcm.Modality == "RTDOSE":
                data["rtdose"] = dcm
            elif dcm.Modality == "RTPLAN":
                data["rtplan"] = dcm
    if "images" in data:
        data["images"].sort(key=compare_dicom_key)
    return data
