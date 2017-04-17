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
    def __init__(self, name="", comment=""):
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
        self.opt_principle = "H2OBased"
        self.opt_alg = "cg"
        self.dose_alg = "cl"
        self.bio_alg = "cl"
        self.eps = 1e-3
        self.geps = 1e-4

        # algorithms for dose optimizations
        self._opt_principles = {"H2Obased": "Very simplified sigle field optimization",
                                "CTbased": "Full optimization, multiple fields."}

        self._opt_methods = {"phys": "Physical dose only",
                             "bio": "Biological optimization"}

        self._opt_algs = {"cl": "classic",
                          "cg": "conjugate graients (default)",
                          "gr": "plain gradients",
                          "bf": "Bortfeld's algorithm",
                          "fr": "Fletcher-Reeves algorithm"}

        self._bio_algs = {"cl": "classic (default)",
                          "ld": "lowdose (faster)"}

        self._dose_alg = {"cl": "classic (default)",
                          "ap": "allpoints",
                          "ms": "multiple scatter"}

        # Scancap parameters:
        # Thickness of ripple filter (0.0 for none)
        # will only used for the TRiP98 Scancap command.
        self.rifi = 0.0  # rifi thickness in [mm]
        self.bolus = 0.0  # amount of bolus to be applied in [mm]
        self.offh2o = 0.0  # Offset of exit-window and ionchambers in [mm]
        self.minparticles = 5000  # smallest amount of particle which will go into a raster spot
        self.scanpath = "none"
        self._scanpaths = {"none:": "No path, output as is",
                           "uw": "U. Weber, efficient",
                           "uw2": "very efficient, works also for non-grid points",
                           "mk": "M. Kraemer, conservative"}

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
        self.incube = ""  # path to DosCube cube for incube optimizations. Important for LET-painting.

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
