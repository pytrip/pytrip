#
#    Copyright (C) 2010-2018 PyTRiP98 Developers.
#
#    This file is part of PyTRiP98.
#
#    PyTRiP98 is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyTRiP98 is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyTRiP98.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Structure:
A Plan() may hold one or multiple Field() objects. When a Plan() is proper setup, it can be executed
with methods in Execute().
"""

import datetime
import os
import uuid
import logging

from pytrip.dos import DosCube
from pytrip.let import LETCube
from pytrip.tripexecuter.execute import Execute
from pytrip.tripexecuter.execparser import ExecParser
from pytrip.tripexecuter.kernel import KernelModel

logger = logging.getLogger(__name__)


class Plan(object):
    """ Class of handling plans.
    """
    # dicts are in the form of
    # "<valid trip tag>": (<enum>, "Short name", "Description, e.g. for alt tags")
    opt_principles = {
        "H2Obased": (0, "Simple opt.", "Very simplified single field optimization"),
        "CTbased": (1, "Full opt.", "Full optimization, multiple fields")
    }

    opt_methods = {
        "phys": (0, "Physical", "Physical dose only [Gy]"),
        "bio": (1, "Biological", "Biological optimization [Gy(RBE)]")
    }

    opt_algs = {
        "cl": (0, "Classic", ""),
        "cg": (1, "Conjugate gradients", "(default)"),
        "gr": (2, "Plain gradients", ""),
        "bf": (3, "Bortfeld", "Bortfeld's algorithm"),
        "fr": (4, "Fletcher-Reeves", "Fletcher-Reeves' algorithm")
    }

    bio_algs = {"cl": (0, "Classic (default)", ""), "ld": (1, "Lowdose (faster)", "")}

    dose_algs = {"cl": (0, "Classic (default)", ""), "ap": (1, "Allpoints", ""), "ms": (2, "Multiple scatter", "")}

    scanpaths = {
        "none": (0, "No path", "output as is"),
        "uw": (1, "U. Weber", "efficient"),
        "uw2": (2, "U. Weber2", "very efficient, works also for non-grid points"),
        "mk": (3, "M. Kraemer", "conservative")
    }

    def __init__(self, default_kernel=KernelModel(), basename="", comment=""):
        """
        A plan Object, which may hold several fields, general setup, and possible also output,
        if it has been calculated.
        :params str basename: TRiP98 qualified plan name. E.g. "test000001"
        :params str comment: any string for documentation purposes.
        :params kernels KernelModel:  a list of Kernel models which describes what projectile to use here.
        So far our versions of TRiP does not support multi-ion optimization.
        Therefore only a single kernel should be provided, i.e. [kernel].
        """

        self.__uuid__ = uuid.uuid4()  # for uniquely identifying this plan
        self.default_kernel = default_kernel
        self.basename = basename.replace(" ", "_")  # TODO: also for Ctx and Vdx, issue when loading DICOMs.
        self.comment = comment

        self.fields = []  # list of Field() objects
        self.voi_target = None  # put target Voi() here. (Not needed if self.target_dose_cube is set (incube))
        self.vois_oar = []  # list of Voi() objects which are considered as OARs

        # results
        self.dosecubes = []  # list of DosCube() objects (i.e. results)
        self.letcubes = []  # list of LETCubes()
        self.out_files = []  # list of files generated which will be returned

        # directories and file paths.
        self.working_dir = ""  # directory where all input files are stored, and where all output files will be put.
        self.temp_dir = ""  # directory where all input files will be temporary stored before execution

        self.hlut_path = "$TRIP98/DATA/19990211.hlut"
        self.dedx_path = "$TRIP98/DATA/DEDX/20040607.dedx"

        self.res_tissue_type = ""
        self.target_tissue_type = ""
        self.active = False
        self.optimize = True  # switch wheter it should be optimized. If False, then a RST file will be loaded instead.
        self.iterations = 500
        self.opt_method = "phys"
        self.opt_principle = "H2Obased"
        self.opt_alg = "cg"
        self.dose_alg = "cl"
        self.bio_alg = "cl"
        self.eps = 1e-3
        self.geps = 1e-4
        self.sis = None  # placeholder for sis parameters

        # Scancap parameters:
        # Thickness of ripple filter (0.0 for none)
        # will only used for the TRiP98 Scancap command.
        self.bolus = 0.0  # amount of bolus to be applied in [mm]
        self.offh2o = 0.0  # Offset of exit-window and ionchambers in [mm]
        self.minparticles = 5000  # smallest amount of particle which will go into a raster spot
        self.scanpath = "none"

        self.random_seed1 = 1000  # Only needed for biological optimization.
        # TODO: TRiP may accept a second "seed2" depending on PRNG. Not implemented here.

        # results requested from optimization
        self.want_phys_dose = True
        self.want_bio_dose = False
        self.want_dlet = False
        self.want_rst = False
        self.want_tlet = False  # TODO: not implemented.

        self.window = []  # window [xmin,xmax,ymin,ymax,zmin,zmax] in CTcoords [mm] for limited dose output.
        self.target_dose = 2.0  # target dose in Gray
        self.target_dose_percent = 100.0  # target dose in percent

        self.target_tissue_type = ""  # target tissue type, for biological optimization
        self.res_tissue_type = ""  # residual tissue types
        self.incube_basename = ""  # To enable incube optimization, set the basename of the cube to be loaded by TRiP

        self._trip_exec = ""  # placeholder for generated TRiP98 .exec commands.
        self._make_sis = ""  # placeholder for generate sistable command

    def __str__(self):
        """ string out handler
        """
        return self._print()

    def _print(self):
        """ Pretty print all attributes
        """
        out = "\n"
        out += "   Plan '{:s}'\n".format(self.basename)
        out += "----------------------------------------------------------------------------\n"
        out += "|  UUID                         : {:s}\n".format(str(self.__uuid__))
        if self.voi_target:
            out += "|  Target VOI                   : {:s}\n".format(self.voi_target.name)
        else:
            out += "|  Target VOI                   : (none set)\n"
        if self.vois_oar:
            for _oar in self.vois_oar:
                out += "|  Organs at Risk VOIs          : {:s}\n".format(_oar.name)
        else:
            out += "|  Organs at Risk VOIs          : (none set)\n"

        if self.fields:
            out += "+---Fields\n"
            for _field in self.fields:
                out += "|   |           #{:d}              : '{:s}'\n".format(_field.number, _field.basename)
        else:
            out += "|   Fields                      : (none set)\n"

        if self.dosecubes:
            out += "+---Dose cubes:\n"
            for _dosecube in self.dosecubes:
                out += "|   |                              : {:s}\n".format(_dosecube.name)
        else:
            out += "|  Dose cubes                   : (none set)\n"

        if self.letcubes:
            out += "+---LET cubes:\n"
            for _letcube in self.letcubes:
                out += "|   |                              : {:s}\n".format(_letcube.name)
        else:
            out += "|  LET cubes                    : (none set)\n"

        out += "|\n"
        out += "| Directories\n"
        out += "|   Working directory           : {:s}\n".format(self.working_dir)
        out += "|   HLUT path                   : {:s}\n".format(self.hlut_path)
        out += "|   dE/dx path                  : {:s}\n".format(self.dedx_path)

        out += "|\n"
        out += "| Sis table generation\n"
        out += "|   {:s}\n".format(self._make_sis)

        out += "|\n"
        out += "| Optimization parameters\n"
        out += "|   Optimization enabled        : {:s}\n".format(str(self.optimize))
        out += "|   Optimization method         : '{:s}' {:s}\n".format(self.opt_method,
                                                                        self.opt_methods[self.opt_method][1])
        out += "|   Optimization principle      : '{:s}' {:s}\n".format(self.opt_principle,
                                                                        self.opt_principles[self.opt_principle][1])
        out += "|   Optimization algorithm      : '{:s}' {:s}\n".format(self.opt_alg, self.opt_algs[self.opt_alg][1])
        out += "|   Dose algorithm              : '{:s}' {:s}\n".format(self.dose_alg, self.dose_algs[self.dose_alg][1])
        out += "|   Biological algorithm        : '{:s}' {:s}\n".format(self.bio_alg, self.bio_algs[self.bio_alg][1])
        out += "|   Iterations                  : {:d}\n".format(self.iterations)
        out += "|   eps                         : {:.2e}\n".format(self.eps)
        out += "|   geps                        : {:.2e}\n".format(self.geps)

        out += "|\n"
        out += "| Scanner capabilities (Scancap)\n"
        out += "|   Bolus thickness             : {:.3f} [mm]\n".format(self.bolus)
        out += "|   H2O offset                  : {:.3f} [mm]\n".format(self.offh2o)
        out += "|   Min particles               : {:d}\n".format(self.minparticles)
        out += "|   Scanpath                    : '{:s} {}'\n".format(self.scanpath, self.scanpaths[self.scanpath])
        out += "|\n"
        out += "| Optimization target\n"
        out += "|   Relative target dose        : {:.1f} %\n".format(self.target_dose_percent)
        out += "|   100.0 % target dose set to  : {:.2f} [Gy]\n".format(self.target_dose)
        if self.incube_basename:
            out += "|   Incube optimization         : {:s}\n".format(self.incube_basename + '.dos')
        else:
            out += "|   Incube optimization         : (none set)\n"

        out += "|\n"
        out += "| Biological parameters\n"
        out += "|   Target tissue type          : '{:s}'\n".format(self.target_tissue_type)
        out += "|   Residual tissue type        : '{:s}'\n".format(self.res_tissue_type)
        out += "|   Random seed #1              : {:d}\n".format(self.random_seed1)

        out += "|\n"
        out += "| Requested output\n"
        out += "|   Physical dose cube          : {:s}\n".format(str(self.want_phys_dose))
        out += "|   Biological dose cube        : {:s}\n".format(str(self.want_bio_dose))
        out += "|   Dose averaged LET cube      : {:s}\n".format(str(self.want_dlet))
        out += "|   Track averaged LET cube     : {:s}\n".format(str(self.want_tlet))
        out += "|   Raster scan files           : {:s}\n".format(str(self.want_rst))
        if self.window:
            out += "|   Cube output window\n"
            out += "|      Xmin / Xmax              : {:.2f} / {:.2f}\n".format(self.window[0], self.window[1])
            out += "|      Ymin / Ymax              : {:.2f} / {:.2f}\n".format(self.window[2], self.window[3])
            out += "|      Zmin / Zmax              : {:.2f} / {:.2f}\n".format(self.window[4], self.window[5])
        else:
            out += "|   Cube output window          : (none set)\n"
        return out

    def save_plan(self, images, path):
        """ Saves the complete plan to path.
        1) *.exec
        2) *.dos
        """
        self.save_exec(path + ".exec")
        self.save_data(images, path)

    def save_exec(self, exec_path=None):
        """ Generates (overwriting) self._trip_exec, and saves it to exec_path.
        """

        # _trip_exec is marked, private, users should not touch it.
        # Here it will be overwritten no matter what.
        self.make_exec()

        if exec_path is None:
            exec_path = os.path.join(self.temp_dir, self.basename, "*.exec")

        with open(exec_path, "w") as f:
            f.write(self._trip_exec)

    def get_exec(self):
        if not self._trip_exec:
            self.make_exec()

        return self._trip_exec

    def read_exec(self, exec_path):
        """ Reads an .exec file onto self.
        """
        _exec = ExecParser(self)
        _exec.parse_exec(exec_path)

    def save_data(self, _images, path):
        """ Save this plan, including associated data.
        TODO: may have to be implemented in a better way.
        """
        # t = Execute(images)
        # t.set_plan(self)
        # t.save_data(path)
        for dos in self.dosecubes:
            dos.write(path + "." + dos.get_type() + DosCube.data_file_extension)

        for let in self.letcubes:
            let.write(path + "." + let.get_type() + LETCube.data_file_extension)

    def load_let(self, path):
        """ Load and append a new LET cube from path to self.letcubes.
        """

        let = LETCube()
        let.read(os.path.splitext(path)[0] + ".dos")

        # TODO: check if cube is already loaded, if so, dont load.
        # UUID for LET cubes would be useful then?
        self.letcubes.append(let)

    def clean_voi_caches(self):
        """ TODO: document me
        """
        for voi in self.vois:
            voi.clean_cache()

    def load_dose(self, path, _type, target_dose=0.0):
        """ Load and append a new DosCube from path to self.doscubes.
        """
        dos = DosCube()
        dos.read(os.path.splitext(path)[0] + DosCube.data_file_extension)
        dos._type = _type
        dos.target_dose = target_dose
        self.dosecubes.append(dos)

    def destroy(self):
        """ Destructor for Vois and Fields in this class.
        """
        self.vois.destroy()
        self.fields.destroy()

    def make_exec(self, no_output=False):
        """
        Generates the .exec script from the plan and stores it into self._trip_exec.
        """

        logger.info("Generating the trip_exec script...")

        # All plans must be the same projectile
        # TODO: throw some error if this is not the case
        # projectile = self.fields[0].projectile

        output = []

        output.extend(self._make_exec_header())  # directories and scancap
        output.extend(self._make_exec_input())  # input data, CT and VDX
        output.extend(self._make_exec_fields())  # setup all fields
        output.extend(self._make_exec_oars())  # define OARs

        # for incube optimization
        if self.incube_basename:
            logger.info("Incube optimization selected by {:s}.dos".format(self.incube_basename))
            output.extend(self._make_exec_plan(incube=self.incube_basename))
        else:
            output.extend(self._make_exec_plan())

        # optimization is optional, alternatively do a forward calculation based on .rst file, or do nothing.
        if self.optimize:
            output.extend(self._make_exec_opt())

        # attach the wanted output:
        output.extend(self._make_exec_output(self.basename, self.fields))

        output.extend(["exit"])
        out = "\n".join(output) + "\n"
        self._trip_exec = out
        return out

    def _make_exec_header(self):
        """
        Add some intro header and timestamp.
        Prepare sis, hlut, ddd, dedx, and scancap commands.
        # TODO: scancap could go out into a new method.
        # TODO: check input params
        :returns: an array of lines, one line per command.
        """

        output = []
        output.append("* {:s} created by PyTRiP98 on the {:s}".format(self.basename + ".exec",
                                                                      str(datetime.datetime.now())))
        # TODO: add user and host
        # We can only check if dir exists, if this is supposed to run locally.

        if hasattr(self, "UUID"):
            output.append("*$$$ Plan UUID {:s}".format(self.UUID))
        output.append("*")
        output.append("time / on")
        output.append("sis  * /delete")
        output.append("hlut * /delete")
        output.append("ddd  * /delete")
        output.append("dedx * /delete")
        output.append('dedx "{:s}" / read'.format(self.dedx_path))
        output.append('hlut "{:s}" / read'.format(self.hlut_path))

        # ddd, spc, and sis:
        output.append('ddd "{:s}" / read'.format(self.default_kernel.ddd_path))

        if self.default_kernel.spc_path:  # False for None and empty string.
            output.append('spc "{:s}" / read'.format(self.default_kernel.spc_path))

        if self.default_kernel.sis_path:
            output.append('sis "{:s}" / read'.format(self.default_kernel.sis_path))
        else:
            if not self._make_sis:
                logger.error("No SIS table loaded or generated.")
                raise Exception("No SIS table loaded or generated.")
            output.append(self._make_sis)

        # Scancap:
        opt = "scancap / offh2o({:.3f})".format(self.offh2o)
        opt += " rifi({:.3f})".format(self.default_kernel.rifi_thickness)  # rifi support is not fully implemented yet
        opt += " bolus({:.3f})".format(self.bolus)
        opt += " minparticles({:d})".format(self.minparticles)
        opt += " path({:s})".format(self.scanpath)
        output.append(opt)

        output.append("random {:d}".format(self.random_seed1))
        return output

    def _make_exec_input(self):
        """ Add CT and target VOI to exec file.
        """

        _name = self.voi_target.name.replace(" ", "_")

        output = []
        output.append("ct \"{:s}\" / read".format(self.basename))
        output.append("voi \"{:s}\" / read select(\"{:s}\")".format(self.basename, _name))
        output.append("voi \"{:s}\" / maxdosefraction({:.3f})".format(_name, self.target_dose_percent * 0.01))
        return output

    def _make_exec_fields(self):
        """ Generate .exec command string for one or more fields.
        if Plan().optimize is False, then it is expected there will
        be a path to a rasterscan file in field.rst_path

        :returns: an array of lines, one line per field in fields.
        """
        fields = self.fields

        output = []
        for i, _field in enumerate(fields):
            if self.optimize:
                # this is a new optimization:
                line = "field {:d} / new".format(i + 1)
            else:
                # or, there is a precalculated raster scan file which will be used instead:
                line = "field {:d} / read file({:s})".format(i + 1, _field.rasterfile_path)

            line += " fwhm({:.3f})".format(_field.fwhm)

            line += " raster({:.2f},{:.2f})".format(_field.raster_step[0], _field.raster_step[1])
            # TODO: convert if Dicom angles were given
            # gantry, couch = angles_to_trip(_field.gantry(), _field.couch())
            line += " couch({:.1f})".format(_field.couch)
            line += " gantry({:.1f})".format(_field.gantry)

            # set isocenter if specified in field
            _ic = _field.isocenter
            if len(_ic) >= 3:
                line += " target({:.1f},{:.1f},{:.1f}) ".format(_ic[0], _ic[1], _ic[2])

            # set dose extension:
            # TODO: check number of decimals which make sense
            # TODO: zeros allowed?
            line += " doseext({:.4f})".format(_field.dose_extension)
            line += " contourext({:.2f})".format(_field.contour_extension)
            line += " zsteps({:.3f})".format(_field.zsteps)
            line += ' proj({:s})'.format(_field.kernel.projectile.trip98_format())
            output.append(line)

        return output

    def _make_exec_oars(self):
        """ Generates the list of TRiP commands specifying the organs at risk.
        :params [str] oar_list: list of VOIs which are organs at risk (OARs)
        """

        output = []
        for oar in self.vois_oar:
            _out = "voi " + oar.name.replace(" ", "_")
            _out += " / maxdosefraction({.3f}) oarset".format(oar.max_dose_fraction)
            output.append(_out)
        return output

    def _make_exec_plan(self, incube=""):
        """ Generates the plan command in TRiP98
        """
        output = []
        plan = "plan / dose({:.2f}) ".format(self.target_dose)
        if self.target_tissue_type:
            plan += "target({:s}) ".format(self.target_tissue_type)
        if self.res_tissue_type:
            plan += "residual({:s}) ".format(self.res_tissue_type)
        if incube != "":
            plan += "incube({:s})".format(self.incube_basename)
        output.append(plan)
        return output

    def _make_exec_opt(self):
        """ Generates the optimizer command for TRiP based on Plan() loaded into self.
        :returns: an array of lines, but only holding a single line containing the entire command.
        """
        # example output:
        # opt / field(*) ctbased bio dosealg(ap) optalg(cg) bioalg(ld) geps(1E-4) eps(1e-3) iter(500)
        #

        opt = "opt / field(*)"

        if self.opt_principle not in self.opt_principles:
            logger.error("Unknown optimization principle {:s}".format(self.plan.opt_principle))
        opt += " {:s}".format(self.opt_principle)  # ctbased or H2Obased

        if self.opt_method not in self.opt_methods:
            logger.error("Unknown optimization method {:s}".format(self.plan.opt_method))
        opt += " {:s}".format(self.opt_method)  # "phys" or "bio"

        if self.dose_alg not in self.dose_algs:
            logger.error("Unknown optimization dose algorithm{:s}".format(self.plan.dose_alg))
        opt += " dosealg({:s})".format(self.dose_alg)  # "ap",..

        if self.opt_alg not in self.opt_algs:
            logger.error("Unknown optimization method {:s}".format(self.plan.opt_alg))
        opt += " optalg({:s})".format(self.opt_alg)  # "cl"...

        # TODO: sanity check numbers
        opt += " geps({:.2e})".format(self.geps)
        opt += " eps({:.2e})".format(self.eps)
        opt += " iter({:d})".format(self.iterations)

        return [opt]

    def _make_exec_output(self, basename, fields):
        """
        Generate TRiP98 exec commands for producing various output.
        :params str basename: basename for output files (i.e. no suffix) and no trailing dot
        :params fields:

        :returns: output string
        """
        output = []
        window = self.window
        window_str = ""
        if len(window) == 6:
            window_str = " window({:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}) ".format(
                window[0],
                window[1],  # Xmin/max
                window[2],
                window[3],  # Ymin/max
                window[4],
                window[5])  # Zmin/max

        if self.want_phys_dose:
            line = 'dose "{:s}."'.format(basename)
            self.out_files.append(basename + ".phys.dos")
            self.out_files.append(basename + ".phys.hed")
            line += ' /calculate  alg({:s})'.format(self.dose_alg)
            line += window_str
            line += ' field(*) write'
            output.append(line)

        if self.want_bio_dose:
            line = 'dose "{:s}."'.format(basename)
            self.out_files.append(basename + ".bio.dos")
            self.out_files.append(basename + ".bio.hed")
            line += ' /calculate  bioalgorithm({:s})'.format(self.bio_alg)
            line += window_str
            line += ' biological norbe field(*) write'
            output.append(line)

        if self.want_dlet:
            line = 'dose "{:s}."'.format(basename)
            self.out_files.append(basename + ".dosemlet.dos")
            self.out_files.append(basename + ".dosemlet.hed")
            line += ' /calculate  alg({:s})'.format(self.dose_alg)
            line += window_str
            line += ' field(*) dosemeanlet write'
            output.append(line)

        if self.want_rst and self.optimize:
            for i, field in enumerate(fields):
                output.append('field {:d} / write file({:s}.rst) reverseorder '.format(i + 1, field.basename))
                self.out_files.append(field.basename + ".rst")

        for i, field in enumerate(fields):
            if field.save_bev_file:
                bev_filename = field.bev_filename
                if not bev_filename:
                    bev_filename = field.basename + ".bev.gd"
                line = "field {:d} /bev(*) file({:s})".format(i + 1, bev_filename)
                output.append(line)

                self.out_files.append(bev_filename)

        # TODO: add various .gd files
        return output

    def _make_exec_inp_data(self, projectile=None):
        """ Prepare CTX, VOI loading and RBE base data

        :returns: an array of lines, one line per command.
        """
        output = []

        output.append("ct {:s} / read".format(self._plan_name))  # loads CTX
        output.append("voi {:s} / read".format(self._plan_name))  # loads VDX
        output.append("voi * /list")  # display all loaded VOIs. Helps for debugging.

        # TODO: consider moving RBE file to DDD/SPC/SIS/RBE set
        # TODO: check rbe.get_rbe_by_name method
        if self.target_tissue_type:  # if not None or empty string:
            rbe = self.rbe.get_rbe_by_name(self.target_tissue_type)
            output.append("rbe '{:s}' / read".format(rbe.path))

        if self.res_tissue_type:
            rbe = self.rbe.get_rbe_by_name(self.target_tissue_type)
            output.append("rbe '{:s}' / read".format(rbe.path))

        return output

    def make_sis(self, projectile, focus=(), intensity=(), position=(), path="", write=False):
        """
        Creates a SIS table based on path, focus, intensity and position.
        example:
        plan.make_sis("1H", "4", "1E6,1E7,1E8", (2, 20, 0.1))
        results in:
        makesis 1H / focus(4) intensity(1E6,1E7,1E8) position(2 TO 20 BY 0.1)

        :params str projectile" such as "12C" or "1H"
        :params str focus: comma-delimited list of FWHM focus positions in [mm], such as "2,4,6,8"
        :params str intensities: tuple/list of intensities, such as "1E6,1E7,1E8"
        :params float position: tuple describing min/max/step range, i.e. (5,40,0.1) for 5 to 40 cm in 1 mm steps.
        :params path: write path
        :params write: Set to True, if the generated sis file should be written to path when executed.
        """

        self._make_sis = "makesis {:s} /".format(projectile)
        if focus:
            self._make_sis += " focus({:s})".format(focus)
        if intensity:
            self._make_sis += " intensity({:s})".format(intensity)
        if position:
            self._make_sis += " position({:.2f} TO {:.2f} BY {:.2f})".format(position[0], position[1], position[2])

        if write:
            self._make_sis += "sis \n{:s} / write".format(path)
