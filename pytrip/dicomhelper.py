import dicom
import os

def compare_dicom_ct(dcm1,dcm2):
	diff = float(dcm1.ImagePositionPatient[2])-float(dcm2.ImagePositionPatient[2])
	if diff > 0:
		return 1
	return -1
def read_dicom_folder(path):
	if os.path.isdir(path) is False:
		raise IOError, "Folder does not exist"
	data = {}
	folder = os.listdir(path)
	for item in folder:
		if os.path.splitext(item)[1] == ".dcm":
			dcm = dicom.read_file(os.path.join(path,item))
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
