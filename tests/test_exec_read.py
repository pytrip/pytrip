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
"""
import os
import unittest
import logging

import pytrip.tripexecuter as pte

import tests.base

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


class TestParseExec(unittest.TestCase):
    """ Tests for pytrip.tripexecuter.execparser
    """
    def setUp(self):
        """ Prepare test environment.
        """
        self.exec_name = "TST003101.exec"

        testdir = tests.base.get_files()
        _exec_dir = os.path.join(testdir, "EXEC")
        self.exec_path = os.path.join(_exec_dir, self.exec_name)

    def test_exec_parse(self):
        """
        """
        logger.info("Test parsing '{:s}'".format(self.exec_path))

        plan = pte.Plan()
        plan.read_exec(self.exec_path)
        print(plan)
        for field in plan.fields:
            print(field)


if __name__ == '__main__':
    unittest.main()
