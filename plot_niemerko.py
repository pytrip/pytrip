import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

import pytrip as pt


dicom_base_dir = "../../Downloads/E2E_V/."

# Dicom dose files:
dicom_RD_fn = ["RD.1.2.246.352.71.7.37402163639.2276256.20230427100959.dcm",
               "RD.1.2.246.352.71.7.37402163639.2276257.20230427100959.dcm",
               "RD.1.2.246.352.71.7.37402163639.2276258.20230427100959.dcm"]

# Dicom LET files (typically also saved as RD*):
dicom_RL_fn = ["RD.1.2.246.352.71.7.37402163639.2276256.20230427100959.dcm",
               "RD.1.2.246.352.71.7.37402163639.2276257.20230427100959.dcm",
               "RD.1.2.246.352.71.7.37402163639.2276258.20230427100959.dcm"]

# structure sets files:
dicom_RS_fn = ["RS.1.2.246.352.71.4.361940808526.1121.20190211110514.dcm"]


vois = ["Brain", "PTV", 'BrainStem', 'Chiasm']  # List of vois to be plotted
cols = ["k", "tab:red", "tab:blue", "tab:orange"]  # list of matplotlib colours
alphs = [0.2, 0.5, 0.8, 0.8]  # list of alphas

# -------------------------------------------------------------------------

# read all fields into dd array
print("OPEN DOSE....")
dd = []
for i, fn in enumerate(dicom_RD_fn):
    _dp = os.path.join(dicom_base_dir, fn)
    print(f"Open {_dp}")
    _dcm = pydicom.read_file(_dp)
    _data = {}
    _data["rtdose"] = _dcm
    _d = pt.DosCube()
    _d.read_dicom(_data)
    dd.append(_d)

# read all LETs into dd array
print("OPEN LET....")
dl = []
for i, fn in enumerate(dicom_RL_fn):
    _dp = os.path.join(dicom_base_dir, fn)
    print(f"Open {_dp}")
    _dcm = pydicom.read_file(_dp)
    _data = {}
    _data["rtdose"] = _dcm
    _d = pt.DosCube()
    _d.read_dicom(_data)
    dl.append(_d)


# read structures
print("OPEN RTSS....")
ds = {}
_sp = os.path.join(dicom_base_dir, dicom_RS_fn[0])
_dcm = pydicom.read_file(_sp)
ds["rtss"] = _dcm

vd = pt.VdxCube(dd[0])
vd.read_dicom(ds)
print(f"Available VOIs: {vd.voi_names()}")

voi_cubes = {}
voi_masks = {}
dose_voxels = {}
let_voxels = {}

print("BUILD VOIs....")
for i, voi in enumerate(vois):
    print(f"    VOI #{i} {voi}")
    _tvoi = vd.get_voi_by_name(voi)
    voi_cubes[voi] = _tvoi.get_voi_cube()
    voi_masks[voi] = (voi_cubes[voi].cube == 1000)

    dose_voxels[voi] = []
    dose_voxels[voi].append(np.sum([d.cube[voi_masks[voi]] for d in dd], axis=0))
    for d in dd:
        dose_voxels[voi].append(d.cube[voi_masks[voi]])

    let_voxels[voi] = []
    # next line is wrong, what is needed is sum_i(d_i * l_i) / (d_i)
    let_voxels[voi].append(np.sum([d.cube[voi_masks[voi]] for d in dd], axis=0))
    for d in dl:
        let_voxels[voi].append(d.cube[voi_masks[voi]])

# -----------------------
# at this point we can access every voxel list for every structure for every field like this:
# dose_voxels["Brain"][0] : all doses for the sum of all 3 fields of the Brain structure
# let_voxels["Brain"][0] : all let for the sum of all 3 fields of the Brain structure

# now do the plotting

plt.xlabel("Total dose in voxel [Gy]")
plt.ylabel("Field #1 dose in voxel [Gy]")
plt.rcParams["legend.markerscale"] = 6.0

for i, voi in enumerate(vois):
    plt.scatter(dose_voxels[voi][0], let_voxels[voi][1], s=1, c=cols[i], alpha=alphs[i], label=voi)
plt.legend(loc='upper left')
plt.xlim([0, 65])
plt.ylim([0, 35])
plt.grid(color='lightgrey')
plt.show()