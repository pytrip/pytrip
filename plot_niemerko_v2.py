import os
import logging
import pydicom
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pytrip as pt

# -------------------------------------------------------------------------

class VoxelData:
    def __init__(self, path_rtss, paths_dose, paths_rqm, vois = []):
        self.foobar = 0
        self.dose = {}
        self.rqm = {}
        self.voi_cubes = {}
        self.voi_masks = {}
        self.voi_names = []

        dd = self.open_RD(paths_dose)
        dl = self.open_RD(paths_rqm)
        vd = self.open_RS(path_rtss, dd)
        self.build_VOIs(dd, dl, vd, vois)

    def open_RD(self, paths_RD):
        """ May be either dose or LET cube"""
        dd = []
        for i, fn in enumerate(paths_RD):
            print(f"Open {fn}")
            _dcm = pydicom.read_file(fn)
            _data = {}
            _data["rtdose"] = _dcm
            _d = pt.DosCube()
            _d.read_dicom(_data)
            dd.append(_d)
        return dd

    def open_RS(self, path_RS, dd):
        print("OPEN RTSS....")
        ds = {}
        _dcm = pydicom.read_file(path_RS)
        ds["rtss"] = _dcm

        vd = pt.VdxCube(dd[0])
        vd.read_dicom(ds)
        return vd

    def build_VOIs(self, dd, dl, vd, vois=[]):

        print("BUILD VOIs....")
        if not vois:
            vois = vd.voi_names()
            print(f"Available VOIs: {vois}")
            self.voi_names = vois
        for i, voi in enumerate(vois):
            print(f"    VOI #{i} {voi}")
            _tvoi = vd.get_voi_by_name(voi)

            self.voi_cubes[voi] = _tvoi.get_voi_cube()
            self.voi_masks[voi] = (self.voi_cubes[voi].cube == 1000)

            # setup dose arrays
            self.dose[voi] = []
            self.dose[voi].append(np.sum([d.cube[self.voi_masks[voi]] for d in dd], axis=0))
            for d in dd:
                self.dose[voi].append(d.cube[self.voi_masks[voi]])

            # setup LET arrays
            self.rqm[voi] = []
            # next line is wrong, what is needed is sum_i(d_i * l_i) / (d_i)
            self.rqm[voi].append(np.sum([d.cube[self.voi_masks[voi]] for d in dl], axis=0))
            for d in dl:
                self.rqm[voi].append(d.cube[self.voi_masks[voi]])


    def save(self, fn=None):
        print(f"Saving {fn}")
        if not fn:
            fn = f"data_niemierko.pkl"
        with open(fn, 'wb') as f:
            pickle.dump(self, f, 2)
        print(f"saved")

    @staticmethod
    def load(fn):
        with open(fn, 'rb') as f:
            return pickle.load(f)



# -----------------------
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

filename_rtss = os.path.join(dicom_base_dir, dicom_RS_fn[0])
filenames_dose = [os.path.join(dicom_base_dir, fn) for fn in dicom_RD_fn]
filenames_let = [os.path.join(dicom_base_dir, fn) for fn in dicom_RL_fn]

v = VoxelData(filename_rtss, filenames_dose, filenames_let)
v.save("my_data.pck")  # consider saving this, loading it is much faster than reprocessing!
# v = VoxelData.load("my_data.pck")  # this is very fast!

# at this point we can access every voxel list for every structure for every field like this:
# v.dose[structure_name][field#] where structure name is string (case insensitive) and field# is the field number, 0 for sum.
#
# v.dose["Brain"][0] # : all doses for the sum of all 3 fields of the Brain structure
# v.rqm["Brain"][0] # : all LET for the sum of all 3 fields of the Brain structure

# now do the plotting

plt.xlabel("Total dose in voxel [Gy]")
plt.ylabel("Field #1 dose in voxel [Gy]")
plt.rcParams["legend.markerscale"] = 6.0

for i, voi in enumerate(vois):
    plt.scatter(v.dose[voi][0], v.rqm[voi][1], s=1, c=cols[i], alpha=alphs[i], label=voi)
plt.legend(loc='upper left')
plt.xlim([0, 65])
plt.ylim([0, 35])
plt.grid(color='lightgrey')
plt.show()