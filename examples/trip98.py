import os
import sys

# necessary pytrip imports
import argparse

from pytrip import VdxCube
from pytrip.ctimage import CTImages
from pytrip.ctx import CtxCube
from pytrip.tripexecuter.field import Field
from pytrip.tripexecuter.rbehandler import RBEHandler

from pytrip.tripexecuter.tripplan import TripPlan
from pytrip.tripexecuter.tripexecuter import TripExecuter


from pytrip.tripexecuter.tripplancollection import TripPlanCollection
from pytrip.tripexecuter.tripvoi import TripVoi
from pytrip.tripexecuter.voi import Voi
from pytrip.tripexecuter.voicollection import VoiCollection


def set_trip98_env_variable(trip_path):
    if not os.path.exists(trip_path):
        print('TRiP98 location [%s] not existing, provide correct one'.format(trip_path))
        return 1
    else:
        os.environ['TRIP98'] = trip_path


def set_trip98_path_variable(trip_path):
    trip_bin_path = os.path.join(trip_path, 'bin', 'TRiP98')
    if not os.path.exists(trip_bin_path):
        print('TRiP98 bin directory [%s] not existing'.format(trip_bin_path))
    path_dirs = os.environ['PATH'].split(':')
    found = False
    for directory in path_dirs:
        if 'TRiP98' in os.listdir(directory):
            found = True
    if not found:
        os.environ['PATH'] += ":{:s}".format(os.path.join(trip_path, 'bin'))


def main(args=sys.argv[1:]):

    description = """
    description
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_path", help='input path', type=str)
    parser.add_argument("working_directory", help='working directory', type=str)
    parser.add_argument("--trip", help='directory containing TRiP98 installation', type=str)
    parsed_args = parser.parse_args(args)

    input_path = parsed_args.input_path
    header_filename, data_filename = CtxCube.parse_path(input_path)
    if header_filename is None or data_filename is None:
        print("input path %s doesn't seem to be correct".format(input_path))
        return 1

    working_directory = parsed_args.working_directory
    if not os.path.exists(working_directory):
        print("working directory %s doesn't exists".format(working_directory))
        return 1

    if 'TRIP98' not in os.environ:
        if not parsed_args.trip:
            print('TRiP98 location unknown, provide --trip option')
            return 1
        else:
            set_trip98_env_variable(parsed_args.trip)

    if parsed_args.trip:
        set_trip98_path_variable(parsed_args.trip)

    patient_name = os.path.split(input_path)[-1]  # basename of CT file is patient_name

    # load CT cube
    c = CtxCube()
    c.read(input_path)

    # load VOIs
    structures = VdxCube("", c)
    structures.read(input_path + ".vdx")
    print(structures.get_voi_names())
    v = structures.get_voi_by_name('GTV')
    target_voi = Voi("GTV_VOI", v)

    # default plan
    plan = TripPlan(name=patient_name)

    # set working directory, output will go there
    plan.set_working_dir(working_directory)

    # add target VOI
    plan.add_voi(target_voi)
    plan.get_vois()[0].target = True

    # add default field, carbon ions
    field = Field("Field 1")
    field.set_projectile("C")
    plan.add_field(field)

    # needed to correctly set offset between VOI and CT
    ct_images = CTImages(c)

    # run TRiP98 optimisation
    executer = TripExecuter(ct_images.get_modified_images(plan))
    executer.execute(plan)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
