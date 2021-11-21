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
TODO: documentation here.
"""
import logging
import os
import tempfile

import numpy as np
import pytest

import pytrip as pt
from pytrip.error import InputError
from pytrip.vdx import create_cube, create_voi_from_cube, create_cylinder, create_sphere

logger = logging.getLogger(__name__)


@pytest.mark.smoke
def test_read_with_ct(ctx_corename, vdx_filename):
    logger.info("Creating CT cube from path " + ctx_corename)
    c = pt.CtxCube()
    c.read(ctx_corename)
    v = pt.VdxCube(c)
    logger.info("Adding VDX from path " + vdx_filename)
    v.read(vdx_filename)

    logger.info("Checking len of voi_names")
    assert len(v.voi_names()) == 2

    logger.info("Checking voi_names")
    assert v.voi_names() == ['target', 'voi_empty']

    logger.info("Checking number of vois")
    assert v.number_of_vois() == 2

    logger.info("Checking Vdx str method")
    logger.info(str(v))

    logger.info("Checking Vdx _write_vdx method")
    fd, outfile = tempfile.mkstemp()
    v._write_vdx(outfile)
    assert os.path.exists(outfile) is True
    logger.info("Checking if output file " + outfile + " is not empty")
    assert os.path.getsize(outfile) > 1
    os.close(fd)  # Windows needs it
    os.remove(outfile)

    logger.info("Checking Vdx write method")
    fd, outfile = tempfile.mkstemp()
    v.write(outfile)
    assert os.path.exists(outfile) is True
    logger.info("Checking if output file " + outfile + " is not empty")
    assert os.path.getsize(outfile) > 1
    os.close(fd)  # Windows needs it
    os.remove(outfile)

    logger.info("Checking if getting non-existend VOI throws an exception")
    with pytest.raises(InputError) as e:
        logger.info("Catching {:s}".format(str(e)))
        v.get_voi_by_name('')

    logger.info("Checking Vdx get_voi_by_name method")
    target_voi = v.get_voi_by_name('target')
    assert target_voi.get_name() == 'target'
    assert target_voi.number_of_slices() == 18

    logger.info("Checking Voi get_3d_polygon method")
    assert target_voi.get_3d_polygon() is not None

    # TODO add some assertions
    target_voi.get_2d_projection_on_basis(basis=((1, 0, 0), (0, 2, 0)))

    # TODO add some assertions
    vc = target_voi.get_voi_cube()
    assert vc.is_compatible(c) is True

    # TODO add some assertions
    target_voi.create_point_tree()


@pytest.mark.smoke
def test_read_solo(vdx_filename):
    logger.info("Checking reading VdxCube without CT cube loaded")
    v = pt.VdxCube()
    v.read(vdx_filename)
    assert v is not None


def test_create_voi_cube(ctx_corename):
    logger.info("Creating CT cube from path " + ctx_corename)
    c = pt.CtxCube()
    c.read(ctx_corename)
    logger.info("Generating and adding cube VOI")
    v = create_cube(c, name="cube1", center=[10, 10, 10], width=4, height=4, depth=8)
    assert v.number_of_slices() == 3


def test_create_voi_cyl(ctx_corename):
    logger.info("Creating CT cube from path " + ctx_corename)
    c = pt.CtxCube()
    c.read(ctx_corename)
    logger.info("Generating and adding cylinder VOI")
    v = create_cylinder(c, name="cube2", center=[10, 10, 10], radius=4, depth=8)
    assert v.number_of_slices() == 3


def test_create_voi_sphere(ctx_corename):
    logger.info("Creating CT cube from path " + ctx_corename)
    c = pt.CtxCube()
    c.read(ctx_corename)
    logger.info("Generating and adding sphere VOI")
    v = create_sphere(c, name="cube3", center=[10, 10, 10], radius=8)
    assert v.number_of_slices() == 6

    logger.info("Checking Voi vdx_string method")
    assert len(v.vdx_string()) > 1

    logger.info("Checking Voi get_slice_at_pos method, non-existent slice")
    assert v.get_slice_at_pos(137) is None

    logger.info("Checking Voi get_slice_at_pos method, good slice")
    s = v.get_slice_at_pos(11)
    assert s is not None

    logger.info("Checking Slice vdx_string method")
    assert len(s.vdx_string()) > 1

    logger.info("Checking Slice number_of_contours method")
    assert s.number_of_contours() == 1

    logger.info("Checking Contour create_dicom_contours method")
    dicom_cont = s.create_dicom_contours(v.cube.create_dicom())
    assert len(dicom_cont) == 1

    contour = s.contours[0]

    logger.info("Checking Contour calculate_center method")
    center, area = contour.calculate_center()
    assert np.isclose(center[0], 10.0)
    assert np.isclose(center[1], 10.0)
    assert np.isclose(center[2], 12.0)  # TODO why 12 ?
    assert area > 100.0

    logger.info("Checking Contour vdx_string method")
    assert len(contour.vdx_string()) > 1

    logger.info("Checking Contour number_of_points method")
    assert contour.number_of_points() == 99

    logger.info("Checking Contour has_childs method")
    assert contour.has_childs() is False  # TODO why doesn't have children ?
    contour.print_child(level=0)

    logger.info("Test of Voi get_min_max method")
    s_min, s_max = v.get_min_max()
    assert s_min is not None
    assert s_max is not None
    assert s_max[2] == 18.0
    assert s_min[2] == 3.0

    logger.info("Subsequent test of Voi get_min_max method, as it modifies the object")
    s_min, s_max = v.get_min_max()
    assert s_min is not None
    assert s_max is not None
    assert s_max[2] == 18.0
    assert s_min[2] == 3.0

    logger.info("Test of Voi get_2d_slice method (sagittal)")
    s2 = v.get_2d_slice(plane=pt.Voi.sagittal, depth=10.0)
    assert s2 is not None

    logger.info("Test of Voi get_2d_slice method (coronal)")
    s3 = v.get_2d_slice(plane=pt.Voi.coronal, depth=5.0)
    assert s3 is not None

    logger.info("Test of Voi get_2d_projection_on_basis method")
    v.get_2d_projection_on_basis(basis=((1, 0, 0), (0, 2, 0)))

    logger.info("Test of Voi create_point_tree method")
    v.create_point_tree()
    assert v.points is not None
    assert len(v.points) == 496

    logger.info("Test of Voi get_row_intersections method")
    isec = v.get_row_intersections(pos=(10, 10, 9))
    assert isec is not None
    assert len(isec) == 2

    logger.info("Test of Voi get_voi_cube method")
    vc = v.get_voi_cube()
    assert vc is not None

    center_pos = v.calculate_center()
    assert np.isclose(center_pos[0], 10.0)


def test_create_voi_from_cube(ctx_corename):
    logger.info("Testing cube from path " + ctx_corename)
    c = pt.CtxCube()
    c.read(ctx_corename)
    logger.info("Generating VOI from cube")
    v = create_voi_from_cube(c, name="cube")
    # TODO check if v was created correctly
    assert v.number_of_slices() == 0
