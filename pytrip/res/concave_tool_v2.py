"""
This code is strongly inspired by:
https://github.com/ellieb/EGSnrc/blob/DICOM-viewer/HEN_HOUSE/gui/dose-viewer/volumes/structure-set-volume.js
written by Elise Badun:
https://github.com/ellieb
"""
import numpy as np
import matplotlib.pyplot as plt
from pytrip import _cntr


def to_indices(mm, pixel_size, offsets, slice_thickness):
    x_off, y_off, z_off = offsets
    return [
        round((mm[0] - x_off) / pixel_size),
        round((mm[1] - y_off) / pixel_size),
        round((mm[2] - z_off) / slice_thickness)
    ]


def initialize_bitmap(plane, cube_size):
    x_size, y_size, z_size = cube_size
    bitmap = None
    if plane == 'Sagittal':
        bitmap = np.zeros((z_size, y_size))
    elif plane == 'Coronal':
        bitmap = np.zeros((z_size, x_size))
    return bitmap


def fill_bitmap(bitmap, intersections_list_filtered, offsets, pixel_size, plane, slice_thickness):
    for intersection in intersections_list_filtered:
        for i in range(1, len(intersection), 2):
            point_0 = to_indices(intersection[i - 1], pixel_size, offsets, slice_thickness)
            point_1 = to_indices(intersection[i], pixel_size, offsets, slice_thickness)
            y = point_0[2]
            x_0, x_1 = None, None
            if plane == 'Sagittal':
                x_0 = point_0[1]
                x_1 = point_1[1]
            elif plane == 'Coronal':
                x_0 = point_0[0]
                x_1 = point_1[0]
            bitmap[y, x_0:x_1] = 1


def get_depth(intersections_list_filtered, plane):
    if plane == 'Sagittal':
        return intersections_list_filtered[0][0][0]
    if plane == 'Coronal':
        return intersections_list_filtered[0][0][1]


def translate_contour_to_mm(contours_indices, depth, offsets, pixel_size, plane, slice_thickness):
    contours = []
    x_offset, y_offset, z_offset = offsets
    for v in contours_indices:
        y = v[:, 1] * slice_thickness + z_offset
        contour = None
        if plane == 'Sagittal':
            x = v[:, 0] * pixel_size + y_offset
            zipped_xy = zip(x, y)
            contour = [[depth, real_y, real_z] for real_y, real_z in zipped_xy]
        elif plane == 'Coronal':
            x = v[:, 0] * pixel_size + x_offset
            zipped_xy = zip(x, y)
            contour = [[real_x, depth, real_z] for real_x, real_z in zipped_xy]
        if contour:
            contours.append(contour)
    return contours


def calculate_contour(bitmap):
    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(bitmap, levels=0)
    paths = cp.collections[0].get_paths()
    contours = [np.array(p.vertices) for p in paths]

    return contours


def calculate_contour_v2(bitmap):
    a, b = bitmap.shape
    x, y = np.meshgrid(np.arange(b), np.arange(a))
    contouring_object = _cntr.Cntr(x, y, bitmap)
    traces = contouring_object.trace(0)
    contours = traces[:len(traces) // 2]
    contours_filtered = [contours[i] for i in range(len(contours)) if
                         i == len(contours) - 1 or i < len(contours) - 1 and contours[i] != contours[i + 1]]
    return contours_filtered


def create_contour(intersections_list_mm, cube_size, offsets, pixel_size, plane, slice_thickness):
    intersections_list_filtered = [i for i in intersections_list_mm if i]
    if not intersections_list_filtered:
        return []

    bitmap = initialize_bitmap(plane, cube_size)
    fill_bitmap(bitmap, intersections_list_filtered, offsets, pixel_size, plane, slice_thickness)

    # contours_indices = calculate_contour(bitmap)

    contours_indices = calculate_contour_v2(bitmap)

    depth = get_depth(intersections_list_filtered, plane)
    contours_mm = translate_contour_to_mm(contours_indices, depth, offsets, pixel_size, plane, slice_thickness)

    return contours_mm
