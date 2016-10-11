"""
    This file is part of PyTRiP.

    PyTRiP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyTRiP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyTRiP.  If not, see <http://www.gnu.org/licenses/>
"""

import os
import unittest
import tempfile
import logging

import pytrip as pt

from pytrip.error import InputError
from pytrip.vdx import create_cube, create_voi_from_cube, create_cylinder, create_sphere
import tests.test_base

logger = logging.getLogger(__name__)


class TestVdx(unittest.TestCase):
    def setUp(self):
        testdir = tests.test_base.get_files()
        self.cube000 = os.path.join(testdir, "tst003000")
        self.vdx = os.path.join(testdir, "tst003000.vdx")
        logger.info("Cube path " + self.cube000)
        logger.info("VDX path " + self.vdx)

    def test_read_with_ct(self):
        logger.info("Creating CT cube from path " + self.cube000)
        c = pt.CtxCube()
        c.read(self.cube000)
        v = pt.VdxCube(c, "")
        logger.info("Adding VDX from path " + self.vdx)
        v.read(self.vdx)

        logger.info("Checking len of get_voi_names")
        self.assertEqual(len(v.get_voi_names()), 2)

        logger.info("Checking get_voi_names")
        self.assertEqual(v.get_voi_names(), ['target', 'voi_empty'])

        logger.info("Checking number of vois")
        self.assertEqual(v.number_of_vois(), 2)

        logger.info("Checking Vdx str method")
        self.assertEqual(str(v), "target&voi_empty")

        logger.info("Checking Vdx write_to_voxel method")
        fd, outfile = tempfile.mkstemp()
        v.write_to_voxel(outfile)
        self.assertTrue(os.path.exists(outfile))
        logger.info("Checking if output file " + outfile + " is not empty")
        self.assertGreater(os.path.getsize(outfile), 1)
        os.close(fd)  # Windows needs it
        os.remove(outfile)

        logger.info("Checking Vdx write method")
        fd, outfile = tempfile.mkstemp()
        v.write(outfile)
        self.assertTrue(os.path.exists(outfile))
        logger.info("Checking if output file " + outfile + " is not empty")
        self.assertGreater(os.path.getsize(outfile), 1)
        os.close(fd)  # Windows needs it
        os.remove(outfile)

        logger.info("Checking if getting non-existend VOI throws an exception")
        self.assertRaises(InputError, v.get_voi_by_name, '')

        logger.info("Checking Vdx get_voi_by_name method")
        target_voi = v.get_voi_by_name('target')
        self.assertEqual(target_voi.get_name(), 'target')
        self.assertEqual(target_voi.get_thickness(), 3)
        self.assertEqual(target_voi.number_of_slices(), 0)  # TODO why there are no slices ?

        logger.info("Checking Voi get_3d_polygon method")
        self.assertIsNotNone(target_voi.get_3d_polygon())

        # TODO check why next is not working ?
        # target_voi.get_2d_projection_on_basis( basis=((1,0,0),(0,2,0)))

        # TODO fix it !
        # vc = target_voi.get_voi_cube()

        # TODO check it
        # target_voi.create_point_tree()

    def test_read_solo(self):
        logger.info("Checking reading VdxCube without CT cube loaded")
        v = pt.VdxCube("")
        v.read(self.vdx)

    def test_create_voi_cube(self):
        logger.info("Creating CT cube from path " + self.cube000)
        c = pt.CtxCube()
        c.read(self.cube000)
        logger.info("Generating and adding cube VOI")
        v = create_cube(c, name="cube1", center=[10, 10, 10], width=4, height=4, depth=8)
        self.assertEqual(v.number_of_slices(), 3)

    def test_create_voi_cyl(self):
        logger.info("Creating CT cube from path " + self.cube000)
        c = pt.CtxCube()
        c.read(self.cube000)
        logger.info("Generating and adding cylinder VOI")
        v = create_cylinder(c, name="cube2", center=[10, 10, 10], radius=4, depth=8)
        self.assertEqual(v.number_of_slices(), 3)

    def test_create_voi_sphere(self):
        logger.info("Creating CT cube from path " + self.cube000)
        c = pt.CtxCube()
        c.read(self.cube000)
        logger.info("Generating and adding sphere VOI")
        v = create_sphere(c, name="cube3", center=[10, 10, 10], radius=8)
        self.assertEqual(v.number_of_slices(), 6)

        logger.info("Checking Voi to_voxel_string method")
        self.assertGreater(len(v.to_voxel_string()), 1)

        logger.info("Checking Voi get_slice_at_pos method, non-existent slice")
        self.assertIsNone(v.get_slice_at_pos(137))

        logger.info("Checking Voi get_slice_at_pos method, good slice")
        s = v.get_slice_at_pos(11)
        self.assertIsNotNone(s)

        logger.info("Checking Slice to_voxel_string method")
        self.assertGreater(len(s.to_voxel_string()), 1)

        logger.info("Checking Slice number_of_contours method")
        self.assertEqual(s.number_of_contours(), 1)

        logger.info("Checking Contour create_dicom_contours method")
        dicom_cont = s.create_dicom_contours()
        self.assertEqual(len(dicom_cont), 1)

        contour = s.contour[0]

        logger.info("Checking Contour calculate_center method")
        center, area = contour.calculate_center()
        self.assertAlmostEqual(center[0], 10.0)
        self.assertAlmostEqual(center[1], 10.0)
        self.assertAlmostEqual(center[2], 12.0)  # TODO why 12 ?
        self.assertGreater(area, 100.0)

        logger.info("Checking Contour to_voxel_string method")
        self.assertGreater(len(contour.to_voxel_string()), 1)

        logger.info("Checking Contour number_of_points method")
        self.assertEqual(contour.number_of_points(), 101)

        logger.info("Checking Contour has_childs method")
        self.assertFalse(contour.has_childs())  # TODO why doesn't have children ?
        contour.print_child(level=0)

        logger.info("Test of Voi get_min_max method")
        s_min, s_max = v.get_min_max()
        self.assertIsNotNone(s_min)
        self.assertIsNotNone(s_max)
        # TODO check why get_min_max returns tuple, not a number
        self.assertEqual(s_max[2], 18.0)
        self.assertEqual(s_min[2], 3.0)

        logger.info("Subsequent test of Voi get_min_max method, as it modifies the object")
        s_min, s_max = v.get_min_max()
        self.assertIsNotNone(s_min)
        self.assertIsNotNone(s_max)
        # TODO check why get_min_max returns tuple, not a number
        self.assertEqual(s_max[2], 18.0)
        self.assertEqual(s_min[2], 3.0)

        logger.info("Test of Voi get_2d_slice method (sagital)")
        s2 = v.get_2d_slice(plane=pt.Voi.sagital, depth=10.0)
        self.assertIsNotNone(s2)

        logger.info("Test of Voi get_2d_slice method (coronal)")
        s3 = v.get_2d_slice(plane=pt.Voi.coronal, depth=5.0)
        self.assertIsNotNone(s3)

        logger.info("Test of Voi get_2d_projection_on_basis method")
        v.get_2d_projection_on_basis(basis=((1, 0, 0), (0, 2, 0)))

        logger.info("Test of Voi create_point_tree method")
        v.create_point_tree()
        self.assertIsNotNone(v.points)
        self.assertEqual(len(v.points), 501)

        logger.info("Test of Voi get_row_intersections method")
        isec = v.get_row_intersections(pos=(10, 10, 9))
        self.assertIsNotNone(isec)
        self.assertEqual(len(isec), 2)

        logger.info("Test of Voi get_voi_cube method")
        vc = v.get_voi_cube()
        self.assertIsNotNone(vc)

        # TODO check why not working
        # center_pos = v.calculate_center()
        # self.assertEqual( center_pos[0], 0)

    def test_create_voi_2(self):
        logger.info("Testing cube from path " + self.cube000)
        c = pt.CtxCube()
        c.read(self.cube000)
        logger.info("Generating VOI from cube")
        v = create_voi_from_cube(c, name="cube4")
        # TODO check if v was created correctly
        self.assertEqual(v.number_of_slices(), 0)


if __name__ == '__main__':
    unittest.main()
