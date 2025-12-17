import sys
import argparse
import os
from pytrip.ctx import CtxCube
from pytrip.vdx import VdxCube
from pytrip import dicomhelper

# DICOM study example: A test study containing CT and RT Structure Set files
# can be downloaded from:
# https://github.com/nbassler/dicomexport/tree/main/res/test_studies/DCPT_headphantom


def process_dicom_study(study_path: str) -> int:
    """Process and mask a DICOM study based on VOI contours.

    Args:
        study_path: Path to directory containing DICOM CT and RT Structure Set files
    Returns:
        Exit code (0 for success)
    """
    # Check if the directory exists
    if not os.path.isdir(study_path):
        print(f"Error: Directory '{study_path}' does not exist.", file=sys.stderr)
        return 1

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
    mask = voi_cube.cube == 1000

    # Set all CT Values inside the BODY VOI to zero HU.
    ctx.cube[mask] = 0

    print("Save masked CT as DICOM")
    ctx.write_dicom("CT.masked.dcm")

    return 0


def main() -> int:
    """Parse command line arguments and process DICOM study."""
    parser = argparse.ArgumentParser(
        description="Process a DICOM study: mask CT values inside VOI contours"
    )
    parser.add_argument(
        "study_path",
        help="Path to directory containing DICOM CT and RT Structure Set files"
    )
    args = parser.parse_args()

    return process_dicom_study(args.study_path)


if __name__ == '__main__':
    sys.exit(main())
