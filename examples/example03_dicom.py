import sys
from pytrip.ctx import CtxCube
from pytrip.vdx import VdxCube
from pytrip import dicomhelper

# point to a directory containing DICOM CT and RT Structure Set files:
study_path = "/home/bassler/Projects/dicomexport/res/test_studies/DCPT_headphantom"


def main(args=None) -> int:

    # Load DICOM study
    study = dicomhelper.read_dicom_dir(study_path)

    # study is a dict, print all items of the dict
    for key, value in study.items():
        print(f"Key: {key}, Type: {type(value)}")

    # Load the CT from the DICOM images
    ctx = CtxCube()
    ctx.read_dicom(study)

    print("Show some dimensions of the CT data:")
    print(ctx.xoffset, ctx.yoffset, ctx.zoffset)
    print(ctx.dimx, ctx.dimy, ctx.dimz)

    # Load the VOIs from the RT Structure Set
    vdx = VdxCube(ctx)
    vdx.read_dicom(study)

    # show all VOI names found in the RTSS
    print(vdx.voi_names())

    # Get the BODY VOI
    voi_body = vdx.get_voi_by_name("BODY")

    # Build a DosCube() object based on the VOI.
    # All voxels inside the VOI holds the value 1000, and 0 elsewhere.
    print("get VOI cube")
    voi_cube = voi_body.get_voi_cube()
    mask = (voi_cube.cube == 1000)

    # Set all CT Values inside the BODY VOI to zero HU.
    ctx.cube[mask] = 0

    print("Save masked CT as DICOM")
    ctx.write_dicom("CT.masked.dcm")

    return 0


if __name__ == '__main__':
    sys.exit(main())
