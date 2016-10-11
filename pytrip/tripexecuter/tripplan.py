from pytrip.tripexecuter.pytripobj import pytripObj
from pytrip.tripexecuter.fieldcollection import FieldCollection
from pytrip.tripexecuter.voicollection import VoiCollection
from pytrip.tripexecuter.tripexecuter import TripExecuter, InputError
from pytrip.dos import DosCube
from pytrip.let import LETCube
from pytrip.tripexecuter.dosecube import DoseCube
import os


class TripPlan(pytripObj):
    def __init__(self, name="", comment=""):
        self.save_fields = ["fields", "vois", "remote", "servername", "username", "password", "working_dir",
                            "ddd_folder", "spc_folder", "sis_file", "res_tissue_type", "target_tissue_type", "name",
                            "comment"]
        self.save_fields.extend(["iterations", "opt_method", "opt_princip", "dose_alg", "bio_alg", "opt_alg", "eps",
                                 "geps", "out_phys_dose", "out_bio_dose", "out_dose_mean_let", "out_field",
                                 "out_mean_let", "dose_percent"])
        self.fields = FieldCollection(self)
        self.vois = VoiCollection(self)
        self.remote = False
        self.servername = ""
        self.username = ""
        self.password = ""
        self.working_dir = "$HOME/tmp"
        self.ddd_folder = ""
        self.spc_folder = ""
        self.sis_file = ""
        self.hlut_file = "$TRIP98/DATA/19990211.hlut"
        self.dedx_file = "$TRIP98/DATA/DEDX/20040607.dedx"
        self.res_tissue_type = ""
        self.target_tissue_type = ""
        self.active = False
        self.name = name
        self.comment = comment
        self.dosecube = None
        self.dosecubes = []
        self.optimize = True
        self.letcube = None
        self.iterations = 500
        self.opt_method = "Phys"
        self.opt_princip = "H2OBased"
        self.dose_alg = "classic"
        self.bio_alg = "classic"
        self.opt_alg = "classic"
        self.eps = 1e-3
        self.geps = 1e-4

        self.out_phys_dose = True
        self.out_bio_dose = False
        self.out_dose_mean_let = False
        self.out_field = False
        self.out_mean_let = False
        self.window = []
        self.dose_percent = {}
        self.target_dose_cube = None

    def Init(self, parent):
        self.parent = parent
        self.vois.Init()
        self.fields.Init()

    def set_optimize(self, optimize):
        self.optimize = optimize

    def get_optimize(self):
        return self.optimize

    def set_window(self, window):
        self.window = window

    def get_window(self):
        return self.window

    def set_target_dose_cube(self, dos):
        self.target_dose_cube = dos

    def get_target_dose_cube(self):
        return self.target_dose_cube

    def set_dose_percent(self, ion, dose_percent):
        if dose_percent == "":
            del self.dose_percent[ion]
        self.dose_percent[ion] = float(dose_percent)

    def get_dose_percent(self, ion):
        if ion in self.dose_percent:
            return self.dose_percent[ion]
        return None

    def get_all_dose_percent(self):
        return self.dose_percent

    def remove_dose_percent(self, ion):
        del self.dose_percent[ion]

    def save_plan(self, images, path):
        self.save_exec(images, path + ".exec")
        self.save_data(images, path + ".exec")

    def save_exec(self, images, path):
        t = TripExecuter(images)
        t.set_plan(self)
        t.save_exec(path)

    def save_data(self, images, path):
        t = TripExecuter(images)
        t.set_plan(self)
        t.save_data(path)
        for dos in self.dosecubes:
            dos.get_dosecube().write(path + "." + dos.get_type() + ".dos")
        if self.letcube is not None:
            self.letcube.write(path + "." + "dosmeanlet.dos")

    def get_projectile(self):
        if len(self.fields):
            return self.fields.get_fields()[0].get_projectile()

    def get_out_phys_dose(self):
        return self.out_phys_dose

    def set_out_phys_dose(self, value):
        self.out_phys_dose = value

    def get_out_bio_dose(self):
        return self.out_bio_dose

    def set_out_bio_dose(self, value):
        self.out_bio_dose = value

    def get_out_dose_mean_let(self):
        return self.out_dose_mean_let

    def set_out_dose_mean_let(self, value):
        self.out_dose_mean_let = value

    def get_out_mean_let(self):
        return self.out_dose_mean_let

    def set_out_mean_let(self, value):
        self.out_dose_mean_let = value

    def get_out_field(self):
        return self.out_field

    def set_out_field(self, value):
        self.out_field = value

    def set_res_tissue_type(self, res_tissue_type):
        self.res_tissue_type = res_tissue_type

    def get_res_tissue_type(self):
        return self.res_tissue_type

    def set_target_tissue_type(self, target_tissue_type):
        self.target_tissue_type = target_tissue_type

    def get_target_tissue_type(self):
        return self.target_tissue_type

    def is_remote(self):
        return self.remote

    def set_remote_state(self, state):
        self.remote = state

    def get_server(self):
        return self.servername

    def set_server(self, server):
        """ Set server name where TRiP98 is installed, if not installed locally."""
        self.servername = server

    def get_username(self):
        """ Get username for login on remote TRiP98 server."""
        return self.username

    def set_username(self, username):
        """ Set username for login on remote TRiP98 server."""
        self.username = username

    def get_password(self):
        """ Get password for login on remote TRiP98 server. Note, this stored unencrypted in cleartext."""
        return self.password

    def set_password(self, password):
        """ Set password for login on remote TRiP98 server. Note, this stored unencrypted in cleartext."""
        self.password = password

    def get_working_dir(self):
        """ Get working directory of TRiP98."""
        return self.working_dir

    def set_working_dir(self, working_dir):
        """ Set working directory of TRiP98."""
        self.working_dir = working_dir

    def get_iterations(self):
        """ Get number of iterations for optimization in TRiP98."""
        return self.iterations

    def get_eps(self):
        return self.eps

    def get_geps(self):
        return self.geps

    def get_bio_algorithm(self):
        return self.bio_alg

    def get_opt_algorithm(self):
        return self.opt_alg

    def get_opt_method(self):
        return self.opt_method

    def set_opt_method(self, method):
        self.opt_method = method

    def get_dose_algorithm(self):
        return self.dose_alg

    def active_dose_change(self, dos):
        self.dosecube = dos

    def load_let(self, path):
        if hasattr(self, "letcube"):
            if self.letcube is not None:
                self.remove_let(self.letcube)
        let = LETCube()
        let.read(os.path.splitext(path)[0] + ".dos")

        self.letcube = let
        return let

    def remove_let(self, let):
        self.letcube = None

    def get_let(self):
        if hasattr(self, "letcube"):
            return self.letcube
        return None

    def get_let_cube(self):
        if self.letcube is not None:
            return self.letcube.cube
        return None

    def get_dose(self, type=""):
        if len(self.dosecubes):
            if type == "":
                return self.dosecube
            for cube in self.dosecubes:
                if cube.get_type() == type:
                    return cube
        return None

    def get_dose_cube(self):
        if self.dosecube is not None:
            return self.dosecube.get_dosecube().cube
        return None

    def remove_dose(self, dos):
        self.dosecubes.remove(dos)
        if self.dosecube is dos:
            self.dosecube = None
        self.clean_cache()

    def remove_dose_by_type(self, type):
        dos = None
        for cube in self.dosecubes:
            if cube.get_type() == type:
                dos = cube
        if dos is not None:
            self.dosecubes.remove(dos)
            self.clean_cache()
        return dos

    def clean_cache(self):
        for voi in self.vois:
            voi.clean_cache()

    def add_dose(self, dos, t=""):
        if type(dos) is DosCube:
            dos = DoseCube(dos, t)
        if hasattr(self, "dosecube"):
            if self.dosecube is not None:
                self.remove_dose_by_type(dos.get_type())
        self.dosecubes.append(dos)

        self.set_active_dose(dos)

    def set_active_dose(self, dos):
        self.dosecube = dos
        self.clean_cache()

    def load_dose(self, path, t, target_dose=0.0):
        dos = DosCube()
        dos.read(os.path.splitext(path)[0] + ".dos")
        d = DoseCube(dos, t)
        d.set_dose(target_dose)
        self.add_dose(d)

    def set_eps(self, value):
        try:
            value = float(value)
            if value < 0:
                raise Exception()
            self.eps = value
        except Exception as e:
            raise InputError("eps shoud be a number Larger than 0" + e)

    def set_geps(self, value):
        try:
            value = float(value)
            if value < 0:
                raise Exception()
            self.geps = value

        except Exception as e:
            raise InputError("geps shoud be a number Larger than 0" + e)

    def set_iterations(self, value):
        try:
            value = int(value)
            if value < 0:
                raise Exception()
            self.iterations = value
        except Exception as e:
            raise InputError("iterations shoud be a number Larger than 0" + e)

    def set_bio_algorithm(self, value):
        self.bio_alg = value

    def set_dose_algorithm(self, value):
        self.dose_alg = value

    def set_opt_algorithm(self, value):
        self.opt_alg = value

    def get_opt_princip(self):
        return self.opt_princip

    def set_opt_princip(self, value):
        self.opt_princip = value

    def set_phys_bio(self, value):
        self.phys_bio = value

    def set_name(self, name):
        if self.parent.get_plan(name) is None:
            self.name = name
            return True
        return False

    def get_name(self):
        return self.name

    def get_vois(self):
        return self.vois

    def get_fields(self):
        return self.fields

    def add_voi(self, voi):
        return self.vois.add_voi(voi)

    def add_field(self, field):
        """ Add a new field to the plan object."""
        return self.fields.add_field(field)

    def remove_field(self, field):
        """ Remove field from the plan object."""
        self.fields.remove(field)

    def get_field_datasource(self):
        return self.fields

    def get_voi_datasource(self):
        return self.vois

    def destroy(self):
        self.vois.destroy()
        self.fields.destroy()

    def set_ddd_folder(self, path):  # TODO: propose to rename "folder" -> "dir"
        """ Set directory containing the depth dose kernels in .ddd format """
        self.ddd_folder = path

    def get_ddd_folder(self):
        """ Get directory containing the depth dose kernels in .ddd format """
        return self.ddd_folder

    def set_dedx_file(self, filename):  # TODO: propose to rename "file" -> "path"
        """ Set path to stopping power tables in .dedx format """
        self.dedx_file = filename

    def get_dedx_file(self, filename):
        """ Get path to stopping power tables in .dedx format """
        return self.dedx_file

    def set_hlut_file(self, filename):
        """ Set path to Hounsfield lookup tables in .hlut format """
        self.hlut_file = filename

    def get_hlut_file(self, filename):
        """ Get path to Hounsfield lookup tables in .hlut format """
        return self.hlut_file

    def get_sis_file(self):
        """ Get path to SIS tables."""
        return self.sis_file

    def set_sis_file(self, path):
        """ Set path to SIS tables."""
        self.sis_file = path

    def set_spc_folder(self, path):
        """ Set directory containing the beam kernel spectrum files in .spc format """
        self.spc_folder = path

    def get_spc_folder(self):
        """ Get directory containing the beam kernel spectrum files in .spc format """
        return self.spc_folder
