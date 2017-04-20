#
#    Copyright (C) 2010-2017 PyTRiP98 Developers.
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

import os
import uuid
import logging

logger = logging.getLogger(__name__)


class Plan():
    """ Class of handling plans.
    """

    _opt_principles = {"H2Obased": "Very simplified single field optimization",
                       "CTbased": "Full optimization, multiple fields"}

    _opt_methods = {"phys": "Physical dose only",
                    "bio": "Biological optimization"}

    _opt_algs = {"cl": "classic",
                 "cg": "conjugate graients (default)",
                 "gr": "plain gradients",
                 "bf": "Bortfeld's algorithm",
                 "fr": "Fletcher-Reeves algorithm"}

    _bio_algs = {"cl": "classic (default)",
                 "ld": "lowdose (faster)"}

    _dose_algs = {"cl": "classic (default)",
                  "ap": "allpoints",
                  "ms": "multiple scatter"}

    _scanpaths = {"none": "No path, output as is",
                  "uw": "U. Weber, efficient",
                  "uw2": "very efficient, works also for non-grid points",
                  "mk": "M. Kraemer, conservative"}

    def __init__(self, basename="", comment=""):
        """
        A plan Object, which may hold several fields, general setup, and possible also output,
        if it has been calculated.
        :params str basename: TRiP98 qualified plan name. E.g. "test000001"
        :params str comment:" any string for documentation purposes.
        """

        self.__uuid__ = uuid.uuid4()  # for uniquely identifying this plan
        self.basename = basename
        self.comment = comment

        self.fields = []  # list of Field() objects
        self.voi_target = None  # put target Voi() here. (Not needed if self.target_dose_cube is set (incube))
        self.vois_oar = []  # list of Voi() objects which are considered as OARs

        self.dosecubes = []  # list of DosCube() objects (i.e. results)
        self.letcubes = []  # list of LETCubes()

        # remote planning
        self.remote = False  # remote or local execution
        self.servername = ""
        self.username = ""
        self.password = ""

        # directories and file paths.
        self.working_dir = ""  # directory where all input files are stored, and where all output files will be put.
        self.ddd_dir = "$TRIP98/DATA/DDD/12C/RF3MM/12C.*"
        self.spc_dir = ""
        self.sis_path = ""
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

        # Scancap parameters:
        # Thickness of ripple filter (0.0 for none)
        # will only used for the TRiP98 Scancap command.
        self.rifi = 0.0  # rifi thickness in [mm]
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
        self.target_dose_cube = None  # pass a DosCube() here, for incube() optimization.

        self.target_tissue_type = ""  # target tissue type, for biological optimization
        self.res_tissue_type = ""  # residual tissue types
        self.incube = False  # enable or disable incube optimizations

    def __str__(self):
        return(self._print())

    def _print(self):
        """ Pretty print all attributes
        """
        out = "\n"
        out += "   Plan '{:s}'\n".format(self.basename)
        out += "----------------------------------------------------------------------------\n"
        out += "|  UUID                         : {:s}\n".format(self.__uuid__)
        out += "|  Target VOI                   : {:s}\n".format(self.voi_target.name)
        if self.vois_oar:
            for _oar in self.vois_oar:
                out += "|  Organs at Risk VOIs          : {:s}\n".format(_oar.name)
        else:
            out += "|  Organs at Risk VOIs          : (none set)\n"

        if self.fields:
            out += "+---Fields\n"
            for _field in self.fields:
                out += "|   |           #{:d}              : {:s}\n".format(_field.number, _field.basename)
        else:
            out += "|  Fields                       : (none set)\n"

        if self.dosecubes:
            out += "+---Dose cubes:\n"
            for _dosecube in self.dosecube:
                out += "|   |                              : {:s}\n".format(_dosecube.name)
        else:
            out += "|  Dose cubes                   : (none set)\n"

        if self.letcubes:
            out += "+---LET cubes:\n"
            for _letcube in self.letcube:
                out += "|   |                              : {:s}\n".format(_letcube.name)
        else:
            out += "|  LET cubes                    : (none set)\n"

        out += "|\n"
        out += "| Remote access\n"
        out += "|   Remote execution            : {:s}\n".format(str(self.remote))
        out += "|   Server                      : '{:s}'\n".format(self.servername)
        out += "|   Username                    : '{:s}'\n".format(self.username)
        out += "|   Password                    : '{:s}'\n".format("*" * len(self.password))

        out += "|\n"
        out += "| Directories\n"
        out += "|   Working directory           : {:s}\n".format(self.working_dir)
        out += "|   DDD directory               : {:s}\n".format(self.ddd_dir)
        out += "|   SPC directory               : {:s}\n".format(self.spc_dir)
        out += "|   SIS path                    : {:s}\n".format(self.sis_path)
        out += "|   HLUT path                   : {:s}\n".format(self.hlut_path)
        out += "|   dE/dx path                  : {:s}\n".format(self.dedx_path)

        out += "|\n"
        out += "| Optimization parameters\n"
        out += "|   Optimization enabled        : {:s}\n".format(str(self.optimize))
        out += "|   Optimization method         : '{:s}' {:s}\n".format(self.opt_method,
                                                                        self._opt_methods[self.opt_method])
        out += "|   Optimization principle      : '{:s}' {:s}\n".format(self.opt_principle,
                                                                        self._opt_principles[self.opt_principle])
        out += "|   Optimization algorithm      : '{:s}' {:s}\n".format(self.opt_alg,
                                                                        self._opt_algs[self.opt_alg])
        out += "|   Dose algorithm              : '{:s}' {:s}\n".format(self.dose_alg,
                                                                        self._dose_algs[self.dose_alg])
        out += "|   Biological algorithm        : '{:s}' {:s}\n".format(self.bio_alg,
                                                                        self._bio_algs[self.bio_alg])
        out += "|   Iterations                  : {:d}\n".format(self.iterations)
        out += "|   eps                         : {:.2e}\n".format(self.eps)
        out += "|   geps                        : {:.2e}\n".format(self.geps)

        out += "|\n"
        out += "| Scanner capabilities (Scancap)\n"
        out += "|   Ripple filter thickness     : {:.3f} [mm]\n".format(self.rifi)
        out += "|   Bolus thickness             : {:.3f} [mm]\n".format(self.bolus)
        out += "|   H2O offset                  : {:.3f} [mm]\n".format(self.offh2o)
        out += "|   Min particles               : {:d}\n".format(self.minparticles)
        out += "|   Scanpath                    : '{:s}'\n".format(self.scanpath,
                                                                   self._scanpaths[self.scanpath])
        out += "|\n"
        out += "| Optimization target\n"
        out += "|   Relative target dose        : {:.1f} %\n".format(self.target_dose_percent)
        out += "|   100.0 % target dose set to  : {:.2f} [Gy]\n".format(self.target_dose)
        out += "|   Incube optimization         : {:s}\n".format(str(self.incube))
        if self.target_dose_cube:
            out += "|   Target dose cube name       : '{:s}'\n".format(self.target_dose_cube.name)
        else:
            out += "|   Target dose cube name       : (none set)\n"

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
            out += "|      Xmin / Xmax              : {:.2f}\n".format(self.window[0], self.window[1])
            out += "|      Ymin / Ymax              : {:.2f}\n".format(self.window[2], self.window[3])
            out += "|      Zmin / Zmax              : {:.2f}\n".format(self.window[4], self.window[5])
        else:
            out += "|   Cube output window          : (none set)\n"
        return(out)

    def save_plan(self, images, path):
        """ Saves the complete plan to path.
        1) *.exec
        2) *.dos
        """
        self.save_exec(images, path + ".exec")
        self.save_data(images, path + ".exec")

    def save_exec(self, images, path):
        from pytrip.tripexecuter2.execute import Execute
        t = Execute(images)
        t.set_plan(self)
        t.save_exec(path)

    def save_data(self, images, path):
        from pytrip.tripexecuter2.execute import Execute
        t = Execute(images)
        t.set_plan(self)
        t.save_data(path)
        for dos in self.dosecubes:
            dos.write(path + "." + dos.get_type() + ".dos")

        for let in self.letcubes:
            let.write(path + "." + dos.get_type() + ".dos")

    def load_let(self, path):
        """ Load and append a new LET cube from path to self.letcubes.
        """
        if hasattr(self, "letcube"):
            if self.letcube is not None:
                self.remove_let(self.letcube)

        from pt import LETCube
        let = LETCube()
        let.read(os.path.splitext(path)[0] + ".dos")

        self.letcubes.append(let)

    def clean_voi_caches(self):
        for voi in self.vois:
            voi.clean_cache()

    def load_dose(self, path, _type, target_dose=0.0):
        """ Load and append a new DOS cube from path to self.doscubes.
        """
        from pt import DosCube
        dos = DosCube()
        dos.read(os.path.splitext(path)[0] + ".dos")
        dos._type = _type
        dos.set_dose(target_dose)
        self.doscubes.append(dos)

    def destroy(self):
        self.vois.destroy()
        self.fields.destroy()
