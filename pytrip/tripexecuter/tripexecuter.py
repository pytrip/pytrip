import os
import shutil
import tarfile
import copy
import uuid

import numpy as np
from subprocess import Popen, PIPE

try:
    import paramiko
except:
    pass

from pytrip.dos import DosCube
from pytrip.let import LETCube
from pytrip.vdx import VdxCube
from pytrip.error import InputError
from pytrip.res.point import get_basis_from_angles, angles_to_trip
import pytriplib


class TripExecuter(object):
    def __init__(self, images, rbe=None):
        self.images = images
        self.rbe = rbe
        self.listeners = []

    def delete_workspace(self):
        shutil.rmtree(self.path)

    def cube_in_other_cube(self, outer, inner):
        m = np.max(inner.cube)
        idx = np.where(inner.cube == m)
        idx = [idx[0][0], idx[1][0], idx[2][0]]
        a = outer.cube[idx[0], idx[1]]
        b = inner.cube[idx[0], idx[1]]
        res = a > b
        if True in res[0:idx[2]] and True in res[idx[2]:-1]:
            return True
        return False

    def analyse_cube(self):
        keys = self.projectiles.keys()

        p1 = self.projectiles[keys[0]]
        p2 = self.projectiles[keys[1]]

        cube1 = p1["target_dos"]
        cube2 = p2["target_dos"]

        if not self.cube_in_other_cube(cube1, cube2):
            temp = p1
            p1 = p2
            p2 = temp
            cube1 = p1["target_dos"]
            cube2 = p2["target_dos"]
            self.execute_order = [keys[0], keys[1]]
            self.split_proj_key = keys[1]

        else:
            self.execute_order = [keys[1], keys[0]]
            self.split_proj_key = keys[0]

        target_cube = copy.deepcopy(cube1)
        shadow_cubes = []
        for i, field in enumerate(p1["fields"]):
            d = DosCube(cube1)

            basis = get_basis_from_angles(field.get_gantry(), field.get_couch())
            basis = basis[0]
            basis = np.array([basis[0] / cube1.pixel_size, basis[1] / cube1.pixel_size,
                              basis[2] / cube1.slice_distance])
            basis /= np.max(np.abs(basis))

            d.cube = pytriplib.create_field_shadow(
                cube1.cube,
                cube2.cube,
                np.array(basis, dtype=np.double))
            target_cube -= d
            shadow_cubes.append(d)

        target_cube.cube[target_cube.cube < 0] = 0
        cube2.cube = cube2.cube + target_cube.cube
        # ~ cube2.cube = pytriplib.extend_cube(cube2.cube)
        cube1.cube = cube1.cube - target_cube.cube
        if len(p1["fields"]) == 2:
            a = self.execute_order.pop(1)
            b = self.projectile_dose_level[a]
            self.execute_order.append(a + str(1))
            self.execute_order.append(a + str(2))
            self.projectile_dose_level[a + str(1)] = b
            self.projectile_dose_level[a + str(2)] = b

    def execute(self, plan, callback=None):
        self.plan = plan
        self.callback = callback
        self.ini_execute()
        if not self.mult_proj:
            self.execute_simple()
        else:
            self.execute_mult_proj()
        if os.path.exists(os.path.join(self.path, self.plan_name) + ".bio.dos") and self.plan.get_out_bio_dose():
            self.plan.load_dose(os.path.join(self.path, self.plan_name) + ".bio.dos", "bio", self.target_dose)
        if os.path.exists(os.path.join(self.path, self.plan_name) + ".phys.dos") and self.plan.get_out_phys_dose():
            self.plan.load_dose(os.path.join(self.path, self.plan_name) + ".phys.dos", "phys", self.target_dose)
        if os.path.exists(os.path.join(self.path, self.plan_name) + ".phys.dos") and self.plan.get_out_dose_mean_let():
            self.plan.load_let(os.path.join(self.path, self.plan_name) + ".dosemlet.dos")
        self.finish()

    def ini_execute(self):
        self.split_plan(self.plan)
        self.plan_name = self.plan.get_name().replace(" ", "_")
        self.working_path = os.path.expandvars(self.plan.get_working_dir())
        if not hasattr(self, "folder_name"):
            self.folder_name = str(uuid.uuid4())
        self.path = os.path.join(self.working_path, self.folder_name)
        self.prepare_folder()
        self.convert_files_to_voxelplan()

    def execute_simple(self):
        self.create_trip_exec_simple()
        self.run_trip()

    def execute_mult_proj(self):
        self.analyse_cube()
        i = 3

        for a in range(i):
            for projectile in self.execute_order:
                self.calculate_rest_dose(projectile)
                if a == i - 1:
                    self.create_trip_exec_mult_proj(projectile, first=(a is 0))
                else:
                    self.create_trip_exec_mult_proj(projectile, last=False, first=(a is 0))
                self.run_trip()

                self.split_fields(projectile)
        self.post_process()

    def create_trip_exec_simple(self, no_output=False):
        oar_list = self.oar_list
        targets = self.targets
        fields = self.plan.get_fields()

        projectile = fields[0].get_projectile()
        output = []
        output.extend(self.create_exec_header())
        output.extend(self.create_exec_load_data_files(projectile))
        output.extend(self.create_exec_field(fields))
        output.extend(self.create_exec_oar(oar_list))
        dosecube = None
        if len(targets) > 1:
            dosecube = DosCube(self.images)
            for i, voi in enumerate(targets):
                temp = DosCube(self.images)
                dose_level = int(voi.get_dose() / self.target_dose * 1000)
                if dose_level == 0:
                    dose_level = -1
                temp.load_from_structure(voi.get_voi().get_voi_data(), dose_level)
                if i == 0:
                    dosecube = temp * 1
                else:
                    dosecube.merge_zero(temp)
            dosecube.cube[dosecube.cube == -1] = int(0)

        if not self.plan.get_target_dose_cube() is None:
            dosecube = self.plan.get_target_dose_cube()

        if dosecube is not None:
            if not no_output:
                dosecube.write(os.path.join(self.path, "target_dose.dos"))
            output.extend(self.create_exec_plan(incube="target_dose"))
        else:
            output.extend(self.create_exec_plan())
        if self.plan.get_optimize():
            output.extend(self.create_exec_opt())

        name = self.plan_name
        output.extend(self.create_exec_output(name, fields))
        out = "\n".join(output) + "\nexit\n"
        self.trip_exec = out

    def create_exec_output(self, name, fields, last=True):
        output = []
        window = self.plan.get_window()
        window_str = ""
        if len(window) is 6:
            window_str = " window(%.2f,%.2f,%.2f,%.2f,%.2f,%.2f) " % (window[0], window[1], window[2], window[3],
                                                                      window[4], window[5])

        if self.plan.get_out_phys_dose() is True:
            output.append('dose "' + name + '." /calculate  alg(' + self.plan.get_dose_algorithm() + ')' + window_str +
                          '  field(*) write')
        if last:
            if self.plan.get_out_bio_dose() is True:
                output.append('dose "' + name + '." /calculate ' + window_str + ' bioalgorithm(' +
                              self.plan.get_bio_algorithm() + ') biological norbe write')
            if self.plan.get_out_dose_mean_let() is True:
                output.append('dose "' + name + '." /calculate ' + window_str + ' field(*) dosemeanlet write')
            if self.plan.get_out_field() is True and self.plan.get_optimize() is True:
                for i, field in enumerate(fields):
                    output.append('field %d /write file(%s.rst) reverseorder ' % (i + 1, field.get_name()))
                    field.set_rasterfile(self.working_path + '//' + self.folder_name + '//' + field.get_name())
        return output

    def create_exec_plan(self, incube=""):
        output = []
        plan = "plan / dose(%.2f) " % self.target_dose
        if not self.plan.get_target_tissue_type() == "":
            plan += "target(%s) " % (self.plan.get_target_tissue_type())
        if not self.plan.get_res_tissue_type() == "":
            plan += "residual(%s) " % (self.plan.get_res_tissue_type())
        if not incube == "":
            plan += "incube(%s) " % incube
        output.append(plan)
        return output

    def create_exec_opt(self):
        output = []
        opt = "opt / field(*) "
        opt += self.plan.get_opt_method() + " "
        opt += "iterations(" + str(self.plan.get_iterations()) + ") "
        opt += "dosealgorithm(" + self.plan.get_dose_algorithm() + ") "
        opt += "" + self.plan.get_opt_princip() + " "
        opt += "geps(" + str(self.plan.get_geps()) + ") "
        opt += "eps(" + str(self.plan.get_eps()) + ") "
        opt += "optalgorithm(" + self.plan.get_opt_algorithm() + ") "
        opt += "bioalgorithm(" + self.plan.get_bio_algorithm() + ") "

        output.append(opt)
        return output

    def create_exec_header(self):
        output = []
        output.append("time / on")
        output.append("sis  * /delete")
        output.append("hlut * /delete")
        output.append("ddd  * /delete")
        output.append("dedx * /delete")
        output.append('dedx "' + self.plan.dedx_file + '" / read')
        output.append('hlut "' + self.plan.hlut_file + '" / read')
        output.append("scancap / offh2o(1.709) rifi(3) bolus(0.000) " "minparticles(5000) path(none)")
        output.append("random 1000")
        return output

    def create_exec_load_data_files(self, projectile):
        output = []
        ddd_folder = self.plan.get_ddd_folder()
        spc_folder = self.plan.get_spc_folder()
        sis_file = self.plan.get_sis_file()
        if len(ddd_folder) > 0:
            output.append('ddd "%s/*" / read' % ddd_folder)
        if len(spc_folder) > 0:
            output.append('spc "%s/*" / read' % spc_folder)
        if len(sis_file) > 0:
            output.append('sis "%s" / read' % sis_file)
        else:
            output.append('sis / dummy')
        if not (len(spc_folder) > 0 or len(ddd_folder) > 0):
            if projectile == "C":
                output.append('ddd "$TRIP98/DATA/DDD/12C/RF3MM/12C*" / read')
                output.append('spc "$TRIP98/DATA/SPC/12C/RF3MM/12C*" / read')
            elif projectile == "H":
                output.append('ddd "$TRIP98/DATA/DDD/1H/RF3MM/1H*" / read')
                output.append('spc "$TRIP98/DATA/SPC/1H/RF3MM/1H*" / read')
            elif projectile == "O":
                output.append('ddd "$TRIP98/DATA/DDD/16O/RF3MM/16O*" / read')
                output.append('spc "$TRIP98/DATA/SPC/16O/RF3MM/16O*" / read')
            elif projectile == "Ne":
                output.append('ddd "$TRIP98/DATA/DDD/20Ne/RF3MM/20Ne*" / read')
                output.append('spc "$TRIP98/DATA/SPC/20Ne/RF3MM/20Ne*" / read')
        output.append("ct " + self.plan_name + " /read")

        output.append("voi " + self.plan_name + "  /read")
        output.append("voi * /list")

        if not self.plan.get_target_tissue_type() == "":
            rbe = self.rbe.get_rbe_by_name(self.plan.get_target_tissue_type())
            output.append("rbe '%s' /read" % (rbe.get_path()))
            if not self.plan.get_res_tissue_type() == "" and \
                    not self.plan.get_res_tissue_type() \
                    == self.plan.get_target_tissue_type():
                rbe = self.rbe.get_rbe_by_name(self.plan.get_target_tissue_type())
                output.append("rbe %s /read" % (rbe.get_path()))

        return output

    def create_exec_field(self, fields):
        output = []
        if self.plan.get_optimize():
            for i, val in enumerate(fields):
                field = "field " + str(i + 1) + " / new "
                field += "fwhm(%d) " % (val.get_fwhm())
                raster = val.get_rasterstep()
                if not raster[0] is 0 and not raster[1] is 0:
                    field += "raster(%.2f,%.2f) " % (raster[0], raster[1])
                gantry, couch = angles_to_trip(val.get_gantry(), val.get_couch())
                field += "couch(" + str(couch) + ") "
                field += "gantry(" + str(gantry) + ") "
                target = val.get_target()
                if len(val.get_target()) is not 0:
                    field += "target(%.1f,%.1f,%.1f) " % (target[0], target[1], target[2])
                if val.get_doseextension() > 0.0001:
                    field += "doseext(" + str(val.get_doseextension()) + ") "
                field += "contourext(" + str(val.get_contourextension()) + ") "
                if val.get_zsteps() > 0.001:
                    field += "zsteps(" + str(val.get_zsteps()) + ") "
                field += 'proj(' + val.get_projectile() + ')'
                output.append(field)
        else:
            for i, val in enumerate(fields):
                field = 'field 1 /read file(' + val.get_rasterfile() + '.rst)'
                field += "fwhm(%d) " % (val.get_fwhm())
                raster = val.get_rasterstep()
                if not raster[0] is 0 and not raster[1] is 0:
                    field += "raster(%.2f,%.2f) " % (raster[0], raster[1])
                gantry, couch = angles_to_trip(val.get_gantry(), val.get_couch())
                field += "couch(" + str(couch) + ") "
                field += "gantry(" + str(gantry) + ") "
                if len(val.get_target()) is not 0:
                    field += "target(" + val.get_target() + ") "
                if val.get_doseextension() > 0.0001:
                    field += "doseext(" + str(val.get_doseextension()) + ") "
                field += "contourext(" + str(val.get_contourextension()) + ") "
                if val.get_zsteps() > 0.001:
                    field += "zsteps(" + str(val.get_zsteps()) + ") "
                field += 'proj(' + val.get_projectile() + ')'
                output.append(field)
        return output

    def create_exec_oar(self, oar_list):
        output = []
        for oar in oar_list:
            output.append("voi " + oar.get_name().replace(" ", "_") + " / maxdosefraction(" + str(
                oar.get_max_dose_fraction()) + ") oarset")
        return output

    def split_fields(self, proj):
        if self.split_proj_key not in self.projectiles.keys():
            return
        name = os.path.join(self.path, self.plan_name) + "_" + self.projectiles[proj]["name"]
        path = os.path.join(name + ".phys.dos")
        temp = DosCube()
        temp.read(path)
        temp.version = "2.0"

        self.rest_dose = self.target_dos - temp

        p = self.projectiles[self.split_proj_key]
        p["target_dos"].cube = pytriplib.extend_cube(p["target_dos"].cube)
        if len(p["fields"]) == 2:
            temp.cube = (temp.cube < self.projectiles[proj]["target_dos"].cube) * \
                self.projectiles[proj]["target_dos"].cube + \
                (temp.cube > self.projectiles[proj]["target_dos"].cube) * temp.cube
            dose = self.target_dos - temp
            field1 = p["fields"][0]
            field2 = p["fields"][1]
            d1 = DosCube(temp)
            d2 = DosCube(temp)
            center = pytriplib.calculate_dose_center(p["target_dos"].cube)

            dose.cube[dose.cube < 0] = 0

            temp.cube *= self.target_dos.cube > 0

            basis = get_basis_from_angles(field1.get_gantry(), field1.get_couch())[0]
            basis = np.array([basis[0] / dose.pixel_size, basis[1] / dose.pixel_size, basis[2] / dose.slice_distance])
            basis = basis / np.max(np.abs(basis)) * .5
            d1.cube = pytriplib.create_field_shadow(
                dose.cube,
                temp.cube,
                np.array(basis, dtype=np.double))
            basis = get_basis_from_angles(field2.get_gantry(), field2.get_couch())[0]
            basis = np.array([basis[0] / dose.pixel_size, basis[1] / dose.pixel_size, basis[2] / dose.slice_distance])
            basis /= np.max(np.abs(basis))
            d2.cube = pytriplib.create_field_shadow(
                dose.cube,
                temp.cube,
                np.array(basis, dtype=np.double))
            a = d2.cube > d1.cube
            b = d2.cube < d1.cube
            d2.cube = p["target_dos"].cube * a
            d1.cube = p["target_dos"].cube * b

            rest = p["target_dos"].cube * ((a + b) < 1)

            b = pytriplib.split_by_plane(rest, center, basis)
            d1.cube += b
            d2.cube += rest - b
            self.plan.add_dose(d1, "H1")
            self.plan.add_dose(d2, "H2")

            self.projectiles[field1.get_projectile() + str(1)] = {
                "target_dos": d1,
                "fields": [field1],
                "name": field1.get_projectile() + str(1),
                "projectile": field1.get_projectile()
            }
            self.projectiles[field2.get_projectile() + str(2)] = {
                "target_dos": d2,
                "fields": [field2],
                "name": field2.get_projectile() + str(2),
                "projectile": field2.get_projectile()
            }
            del self.projectiles[self.split_proj_key]

    def calculate_rest_dose(self, proj):
        self.rest_dose = copy.deepcopy(self.target_dos)
        for k, projectile in self.projectiles.items():
            if k == proj:
                continue
            name = os.path.join(self.path, self.plan_name) + "_" + projectile["name"]
            path = os.path.join(name + ".phys.dos")
            if os.path.exists(path):
                temp = DosCube()
                temp.read(path)
                temp = temp
                self.rest_dose.cube = self.rest_dose.cube - temp.cube
                # ~ self.rest_dose.cube[self.rest_dose.cube<0] = 0

    def post_process(self):
        phys_dose = None
        bio_dose = None
        dose_mean_let = None
        temp_dos = None
        for projectile in self.projectiles:
            name = os.path.join(self.path, self.plan_name) + "_" + self.projectiles[projectile]["name"]
            factor = float(self.projectile_dose_level[projectile]) / 1000
            factor = 1
            if os.path.exists(name + ".bio.dos") and self.plan.get_out_bio_dose():
                path = os.path.join(name + ".bio.dos")
                temp = DosCube()
                temp.read(path)
                temp *= factor
                if self.mult_proj:
                    self.plan.add_dose(temp, "bio_%s" % projectile)
                if bio_dose is None:
                    bio_dose = temp
                else:
                    bio_dose += temp

            if os.path.exists(name + ".phys.dos") and self.plan.get_out_phys_dose():
                path = os.path.join(name + ".phys.dos")
                temp_dos = DosCube()
                temp_dos.read(path)
                temp_dos *= factor
                if self.mult_proj:
                    self.plan.add_dose(temp_dos, "phys_%s" % projectile)
                if phys_dose is None:
                    phys_dose = temp_dos
                else:
                    phys_dose += temp_dos
            if os.path.exists(name + ".dosemlet.dos") and self.plan.get_out_dose_mean_let():

                path = os.path.join(name + ".dosemlet.dos")
                temp = LETCube()
                temp.read(path)
                if not self.mult_proj:
                    dose_mean_let = temp
                else:
                    if dose_mean_let is None:
                        dose_mean_let = temp * temp_dos
                    else:
                        dose_mean_let = dose_mean_let + temp * temp_dos
        out_path = os.path.join(self.path, self.plan_name)
        if bio_dose is not None:
            bio_dose.write(out_path + ".bio.dos")
        if phys_dose is not None:
            phys_dose.write(out_path + ".phys.dos")
        if dose_mean_let is not None:
            if self.mult_proj:
                dose_mean_let /= phys_dose

            dose_mean_let.write(out_path + ".hed")

    def split_plan(self, plan=None):
        self.targets = []
        self.oar_list = []

        dose = 0
        for voi in self.plan.get_vois():
            if voi.is_oar():
                self.oar_list.append(voi)
            if voi.is_target():
                self.targets.append(voi)
                if voi.get_dose() > dose:
                    dose = voi.get_dose()
        if not len(self.targets):
            raise InputError("No targets")
        if not len(self.plan.get_fields()):
            raise InputError("No fields")
        self.target_dose = dose
        if plan is None:
            plan = self.plan
        proj = []
        self.projectile_dose_level = {}
        for field in plan.fields:
            if field.get_projectile() not in proj:
                proj.append(field.get_projectile())
                self.projectile_dose_level[field.get_projectile()] = 0

        if len(proj) > 1:
            self.mult_proj = True
        else:
            self.mult_proj = False

        if self.mult_proj:
            self.projectiles = {}
            for field in plan.fields:
                if field.get_projectile() not in self.projectiles.keys():
                    self.projectiles[field.get_projectile()] = {
                        "target_dos": DosCube(self.images),
                        "fields": [field],
                        "name": field.get_projectile(),
                        "projectile": field.get_projectile()
                    }
                else:
                    self.projectiles[field.get_projectile()]["fields"].append(field)

            self.target_dos = DosCube(self.images)

            for i, voi in enumerate(self.targets):
                temp = DosCube(self.images)
                voi_dose_level = int(voi.get_dose() / dose * 1000)
                temp.load_from_structure(voi.get_voi().get_voi_data(), 1)
                for projectile, data in self.projectiles.items():
                    dose_percent = self.plan.get_dose_percent(projectile)
                    if not voi.get_dose_percent(projectile) is None:
                        dose_percent = voi.get_dose_percent(projectile)
                    proj_dose_lvl = int(voi.get_dose() / self.target_dose * dose_percent * 10)
                    if self.projectile_dose_level[projectile] < proj_dose_lvl:
                        self.projectile_dose_level[projectile] = proj_dose_lvl
                    if proj_dose_lvl == 0:
                        proj_dose_lvl = -1
                    if i == 0:
                        data["target_dos"] = temp * proj_dose_lvl
                    else:
                        data["target_dos"].merge_zero(temp * proj_dose_lvl)
                if i == 0:
                    self.target_dos = temp * voi_dose_level
                else:
                    self.target_dos.merge_zero(temp * voi_dose_level)
            for projectile, data in self.projectiles.items():
                data["target_dos"].cube[data["target_dos"].cube == -1] = int(0)
                self.plan.add_dose(data["target_dos"], "target_%s" % projectile)
            self.rest_dose = copy.deepcopy(self.target_dos)

    def add_log_listener(self, listener):
        self.listeners.append(listener)

    def log(self, txt):
        txt = txt.replace("\n", "")
        for l in self.listeners:
            l.write(txt)

    def prepare_folder(self):
        self.filepath = self.path + self.plan_name
        if os.path.exists(self.path):
            pass
            # shutil.rmtree(self.path)
        else:
            os.makedirs(self.path)

    def run_trip(self):
        if self.plan.remote:
            self.run_trip_remote()
        else:
            self.run_trip_local()

    def run_trip_remote(self):
        self.create_remote_run_file()
        self.compress_files()

        self.copy_files_to_server()

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.plan.get_server(), username=self.plan.get_username(), password=self.plan.get_password())
        commands = ["tar -zxvf temp.tar.gz", "cd %s;bash run" % self.folder_name,
                    "tar -zcvf temp.tar.gz %s" % self.folder_name, "rm -r %s" % self.folder_name]
        for cmd in commands:
            self.log(cmd)
            stdin, stdout, stderr = ssh.exec_command(cmd)
            self.log(stdout.read())
        ssh.close()
        self.copy_back_from_server()
        self.decompress_data()

    def run_trip_local(self):
        os.chdir("%s" % (self.path))
        p = Popen(["TRiP98"], stdout=PIPE, stdin=PIPE)

        p.stdin.write(self.trip_exec)
        p.stdin.flush()
        while (True):
            retcode = p.poll()  # returns None while subprocess is running
            line = p.stdout.readline()
            self.log(line)
            if (retcode is not None):
                break
        os.chdir("..")

    def finish(self):
        pass

    def create_trip_exec_mult_proj(self, projectile, no_output=False, last=True, first=True):

        fields = self.projectiles[projectile]["fields"]
        oar_list = self.oar_list

        output = []
        output.extend(self.create_exec_header())
        output.extend(self.create_exec_load_data_files(self.projectiles[projectile]["projectile"]))
        output.extend(self.create_exec_field(fields))
        output.extend(self.create_exec_oar(oar_list))

        dosecube = copy.deepcopy(self.projectiles[projectile]["target_dos"])

        if not no_output:
            if hasattr(self, "rest_dose"):
                dosecube.cube = np.array(
                    (self.rest_dose.cube >= dosecube.cube) * dosecube.cube +
                    (self.rest_dose.cube < dosecube.cube) * self.rest_dose.cube,
                    dtype=np.int16)
                dosecube.cube[dosecube.cube < 0] = 0
                # ~ self.plan.add_dose(self.rest_dose,"rest")
                if not first:
                    a = (self.rest_dose.cube - dosecube.cube) * (dosecube.cube > 0)
                    dosecube.cube += a * dosecube.cube / 500

            dosecube.write(os.path.join(self.path, "target_dose_%s.dos" % self.projectiles[projectile]["name"]))

        self.projectile_dose_level[projectile] = np.max(dosecube.cube)
        output.extend(self.create_exec_plan(incube="target_dose_%s" % self.projectiles[projectile]["name"]))
        if self.plan.get_optimize() is True:
            output.extend(self.create_exec_opt())

        name = self.plan_name + "_" + self.projectiles[projectile]["name"]
        output.extend(self.create_exec_output(name, fields))
        out = "\n".join(output) + "\nexit\n"
        self.trip_exec = out

    def convert_files_to_voxelplan(self):
        out_path = os.path.join(self.path, self.plan_name)
        ctx = self.images
        ctx.patient_name = self.plan_name
        ctx.write(os.path.join(out_path + ".ctx"))
        structures = VdxCube("", ctx)
        structures.version = "2.0"
        liste = []
        area = 0
        for voi in self.plan.get_vois():
            voxelplan_voi = voi.get_voi().get_voi_data()

            if voi.is_target():
                mn, mx = voxelplan_voi.get_min_max()
                area_temp = (mx[0] - mn[0]) * (mx[1] - mn[1]) * (mx[2] - mn[2])
                if area_temp > area:
                    area = area_temp
                    liste.insert(0, voxelplan_voi)
                else:
                    liste.append(voxelplan_voi)
                voxelplan_voi.type = '1'
            else:
                liste.append(voxelplan_voi)
                voxelplan_voi.type = '0'
        for voxelplan_voi in liste:
            structures.add_voi(voxelplan_voi)
        structures.write_to_trip(out_path + ".vdx")

    def compress_files(self):
        self.save_exec(os.path.join(self.working_path, self.folder_name, "plan.exec"))
        tar = tarfile.open(os.path.join(self.working_path, self.folder_name + ".tar.gz"), "w:gz")
        tar.add(self.path, arcname=self.folder_name)
        tar.close()

    def create_remote_run_file(self):
        with open(os.path.join(self.working_path, self.folder_name, "run"), "wb+") as fp:
            fp.write("source ~/.profile\n")
            fp.write("TRiP98 < plan.exec")

    def set_plan(self, plan):
        self.plan = plan
        self.plan_name = self.plan.get_name().replace(" ", "_")

    def save_exec(self, path):
        self.split_plan()
        path1 = path.replace(".exec", "")
        for projectile in self.projectiles:
            self.create_trip_exec(projectile, True)
            with open(path1 + "_" + projectile + ".exec", "wb+") as fp:
                fp.write(self.trip_exec)

    def save_data(self, path):
        out_path = path
        ctx = self.images
        ctx.patient_name = self.plan_name
        ctx.write(os.path.join(out_path + ".ctx"))
        structures = VdxCube("", ctx)
        structures.version = "2.0"
        for voi in self.plan.get_vois():
            voxelplan_voi = voi.get_voi().get_voi_data()
            structures.add_voi(voxelplan_voi)
            if voi.is_target():
                voxelplan_voi.type = '1'
            else:
                voxelplan_voi.type = '0'
        structures.write_to_trip(out_path + ".vdx")

    def get_transport(self):
        transport = paramiko.Transport((self.plan.get_server(), 22))
        transport.connect(username=self.plan.get_username(), password=self.plan.get_password())
        return transport

    def copy_files_to_server(self):
        transport = self.get_transport()
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.put(self.path + ".tar.gz", 'temp.tar.gz')
        sftp.close()
        transport.close()

    def run_ssh_command(self, cmd):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.plan.get_server(), username=self.plan.get_username(), password=self.plan.get_password())
        self.parent.write_to_log("Run Trip\n")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        ssh.close()

    def copy_back_from_server(self):
        transport = self.get_transport()
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get('temp.tar.gz', self.path + ".tar.gz")
        sftp.close()
        transport.close()

    def decompress_data(self):
        output_folder = self.path
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        tar = tarfile.open(self.path + ".tar.gz", "r:gz")
        tar.extractall(self.working_path)

    def clean_up(self):
        f = "%s" % (self.path + ".tar.gz")
        if os.path.exists(f):
            os.remove(f)
        shutil.rmtree("%s" % self.path)

    def visualize_data(self):
        pass
