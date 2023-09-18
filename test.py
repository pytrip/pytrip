import pytrip as pt
import matplotlib.pyplot as plt

p = "../../Downloads/E2E_V/."
pc = "../../Downloads/E2E_V/CT/."
pd = "../../Downloads/E2E_V/RD/."
ps = "../../Downloads/E2E_V/RS/."

my_target_voi = "PTV"

d = pt.dicomhelper.read_dicom_dir(p)
dc = pt.dicomhelper.read_dicom_dir(pc)
dd = pt.dicomhelper.read_dicom_dir(pd)
ds = pt.dicomhelper.read_dicom_dir(ps)

# print("READ CT")
# c = pt.CtxCube()
# c.read_dicom(dc)

# print("READ VDX-CT")
# vc = pt.VdxCube(c)
# vc.read_dicom(ds)

# target_voi = vc.get_voi_by_name(my_target_voi)
# voi_cube = target_voi.get_voi_cube()
# mask = (voi_cube.cube == 1000)
# c.cube[mask] = 0

# plt.imshow(mask[200,:,:])
# plt.show()

print("READ DOS")
d = pt.DosCube()
d.read_dicom(dd)

print("READ VDX-dose")
vd = pt.VdxCube(d)
vd.read_dicom(ds)

target_voi = vd.get_voi_by_name(my_target_voi)
voi_cube = target_voi.get_voi_cube()
mask = (voi_cube.cube == 1000)
d.cube[mask] = 0

plt.imshow(mask[33,:,:])
# plt.imshow(d.cube[110,:,:])
plt.colorbar()
plt.show()