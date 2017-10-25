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
Tests for models/*.py
"""
import os
import unittest
import logging
import numpy as np

import pytrip as pt
from pytrip.models.proton import rbe_carabe
from pytrip.models.proton import rbe_wedenberg
from pytrip.models.proton import rbe_mcnamara
from pytrip.models.rcr import sf_rcr

import tests.base

logger = logging.getLogger(__name__)


class TestProton(unittest.TestCase):
    """ Test the proton.py models
    """
    def setUp(self):
        """ Prepare files for tests
        """
        testdir = tests.base.get_files()
        self.cube001 = os.path.join(testdir, "tst003001")
        self.vdx = os.path.join(testdir, "tst003000.vdx")
        logger.info("Cube path " + self.cube001)
        logger.info("VDX path " + self.vdx)
        pass

    def test_carabe(self):
        """ Check if we are able to calculate a few RBE values
        """
        # increasing LET should increase RBE
        rbe1 = rbe_carabe(10.0, 10.0, 5.0)
        rbe2 = rbe_carabe(10.0, 17.0, 5.0)
        self.assertGreater(rbe2, rbe1)
        self.assertGreater(rbe2, 1.0)  # Carabe can actually return values below 1.0 for RBE
        self.assertGreater(10.0, rbe2)  # Sanity check

    def test_wedenberg(self):
        """ Check if we are able to calculate a few RBE values
        """
        # increasing LET should increase RBE
        rbe1 = rbe_wedenberg(10.0, 10.0, 5.0)
        rbe2 = rbe_wedenberg(10.0, 17.0, 5.0)
        self.assertGreater(rbe2, rbe1)
        self.assertGreater(rbe2, 1.0)
        self.assertGreater(10.0, rbe2)  # Sanity check

    def test_mcnamara(self):
        """ Check if we are able to calculate a few RBE values
        """
        # increasing LET should increase RBE
        rbe1 = rbe_mcnamara(10.0, 10.0, 5.0)
        rbe2 = rbe_mcnamara(10.0, 17.0, 5.0)
        self.assertGreater(rbe2, rbe1)
        self.assertGreater(rbe2, 1.0)
        self.assertGreater(10.0, rbe2)  # Sanity check

    def test_mcnamara_cube(self):
        """ McNamara test on real cubes.
        """
        dose = pt.DosCube()
        dose.read(self.cube001)
        let = pt.LETCube()
        let.read(self.cube001)
        v = pt.VdxCube(dose)
        logger.info("Adding VDX from path " + self.vdx)
        v.read(self.vdx)

        # increasing LET should increase RBE
        abx = 10.0  # alpha/beta ratio for x-rays [Gy]
        rbe1 = rbe_mcnamara(dose.cube, let.cube, abx)
        rbe2 = rbe_mcnamara(dose.cube, let.cube, 2.0)

        self.assertTrue(np.all(rbe2 >= rbe1))  # RBE goes up as abx goes down.


class TestRCR(unittest.TestCase):
    """ Test the rcr.py model
    """
    def setUp(self):
        """ Prepare files for tests
        """
        pass

    def test_rcr(self):
        """ Test RCR model
        """
        dose = 2.0  # Gy
        let = 50.0  # keV/um

        sf1 = sf_rcr(dose, let)
        # Survival should always be 0.0 <= sf <= 1.0
        self.assertGreater(1.0, sf1)
        self.assertGreater(sf1, 0.0)

        # add hypoxia, should increase survival
        sf2 = sf_rcr(dose, let, 10.0)  # some oxygenation -> less survival
        self.assertGreater(1.0, sf2)
        self.assertGreater(sf2, 0.0)

        sf1 = sf_rcr(dose, let, 0.0)  # no oxygenation -> much survival
        self.assertGreater(1.0, sf2)
        self.assertGreater(sf2, 0.0)
        self.assertGreater(sf1, sf2)

        # increase LET, should reduce survival
        sf2 = sf_rcr(dose, let + 10.0)
        self.assertGreater(1.0, sf2)
        self.assertGreater(sf2, 0.0)
        self.assertGreater(sf1, sf2)


if __name__ == '__main__':
    unittest.main()
