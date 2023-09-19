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

print("READ CT")
c = pt.CtxCube()
c.read_dicom(dc)

print("READ VDX-CT")
vc = pt.VdxCube(c)
vc.read_dicom(ds)

target_voi = vc.get_voi_by_name(my_target_voi)
voi_cube = target_voi.get_voi_cube()
mask = (voi_cube.cube == 1000)
c.cube[mask] = 0

plt.imshow(c.cube[100,:,:])
plt.show()

print("READ DOS")
d = pt.DosCube()
d.read_dicom(dd)

print(d.dimx, d.dimy)

print("READ VDX-dose")
vd = pt.VdxCube(d)
vd.read_dicom(ds)

print(d.cube.shape)
print(vd.cube.cube.shape)

target_voi = vd.get_voi_by_name(my_target_voi)
voi_cube = target_voi.get_voi_cube()
print(voi_cube.cube.shape)
mask = (voi_cube.cube == 1000)
print(mask.shape)

# exit()
d.cube[mask] = 7

# for i in range(d.dimz):
#     print(i, d.cube[i,:,:].max())

plt.imshow(d.cube[60,:,:])
plt.colorbar()
plt.show()