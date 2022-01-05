"""
This code is strongly inspired by:
https://github.com/ellieb/EGSnrc/blob/DICOM-viewer/HEN_HOUSE/gui/dose-viewer/volumes/structure-set-volume.js
written by Elise Badun:
https://github.com/ellieb
"""
import numpy as np
from pytrip import _cntr

sagittal = 2  #: id for sagittal view
coronal = 1  #: id for coronal view


def mm_to_indices(mm, pixel_size, offsets, slice_thickness):
    x_off, y_off, z_off = offsets
    return [
        int(round((mm[0] - x_off) / pixel_size)),
        int(round((mm[1] - y_off) / pixel_size)),
        int(round((mm[2] - z_off) / slice_thickness))
    ]


def initialize_bitmap(plane, cube_size):
    x_size, y_size, z_size = cube_size
    bitmap = None
    if plane == sagittal:
        bitmap = np.zeros((z_size, y_size))
    elif plane == coronal:
        bitmap = np.zeros((z_size, x_size))
    return bitmap


def fill_bitmap(bitmap, intersections_list_filtered, offsets, pixel_size, plane, slice_thickness):
    for intersection in intersections_list_filtered:
        # take each subsequent pair od points
        for i in range(1, len(intersection), 2):
            # translate coordinate in millimeters to indices
            point_0 = mm_to_indices(intersection[i - 1], pixel_size, offsets, slice_thickness)
            point_1 = mm_to_indices(intersection[i], pixel_size, offsets, slice_thickness)
            # so called 'y' is the z index, the number of row to fill
            y = point_0[2]
            # depending on plane, extract boundaries of columns to fill
            x_0, x_1 = None, None
            if plane == sagittal:
                x_0 = point_0[1]
                x_1 = point_1[1]
            elif plane == coronal:
                x_0 = point_0[0]
                x_1 = point_1[0]
            # fill proper row and columns with ones
            bitmap[y, x_0:x_1] = 1


def get_depth(intersections_list_filtered, plane):
    # depth is the same for each point in each list
    if plane == sagittal:
        return intersections_list_filtered[0][0][0]  # value of x coordinate
    if plane == coronal:
        return intersections_list_filtered[0][0][1]  # value of y coordinate
    return None


def translate_contour_to_mm(contours_indices, depth, offsets, pixel_size, plane, slice_thickness):
    x_offset, y_offset, z_offset = offsets
    contours_mm = []
    for v in contours_indices:
        y = v[:, 1] * slice_thickness + z_offset
        contour_mm = None
        if plane == sagittal:
            x = v[:, 0] * pixel_size + y_offset
            zipped_xy = zip(x, y)
            contour_mm = [[depth, real_y, real_z] for real_y, real_z in zipped_xy]
        elif plane == coronal:
            x = v[:, 0] * pixel_size + x_offset
            zipped_xy = zip(x, y)
            contour_mm = [[real_x, depth, real_z] for real_x, real_z in zipped_xy]
        if contour_mm:
            contours_mm.append(contour_mm)
    return contours_mm


def filter_predicate(i, contour):
    is_last = (i == len(contour) - 1)
    return is_last or (not is_last and (contour[i] != contour[i + 1]).any())


def calculate_contour(bitmap):
    # prepare mesh grid for contouring object
    a, b = bitmap.shape
    x, y = np.meshgrid(np.arange(b), np.arange(a))
    # create contouring object using efficient C code
    contouring_object = _cntr.Cntr(x, y, bitmap)
    # trace one and only one isoline
    traces = contouring_object.trace(0)
    # that one isoline can have multiple contours, that are stored in the first half of returned array
    contours = traces[:len(traces) // 2]
    # each contour may have duplicated adjacent vertices, we need to filter them out
    contours_filtered = []
    # for each contour remove adjacent duplicates
    for contour in contours:
        without_duplicates = np.array([contour[i] for i in range(len(contour)) if filter_predicate(i, contour)])
        contours_filtered.append(without_duplicates)

    return contours_filtered


def create_contour(intersections_list_mm, cube_size, offsets, pixel_size, plane, slice_thickness):
    # filter empty intersections
    intersections_list_filtered = [item for item in intersections_list_mm if item]
    # return empty list if filtered list is empty
    if not intersections_list_filtered:
        return []
    # create proper size bitmap filled with zeros
    bitmap = initialize_bitmap(plane, cube_size)
    # fill the bitmap with ones in proper positions
    fill_bitmap(bitmap, intersections_list_filtered, offsets, pixel_size, plane, slice_thickness)
    # calculate contour with vertices in terms of indices in bitmap
    contours_indices = calculate_contour(bitmap)
    # get depth for which contour is calculated
    depth = get_depth(intersections_list_filtered, plane)
    # translate vertices back to millimeters
    contours_mm = translate_contour_to_mm(contours_indices, depth, offsets, pixel_size, plane, slice_thickness)

    return contours_mm
