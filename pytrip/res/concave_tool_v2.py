import numpy as np
import matplotlib.pyplot as plt


def to_ind(mm, pix_size, x_off, y_off, z_off, s_thick):
    return [
        int((mm[0] - x_off) / pix_size),
        int((mm[1] - y_off) / pix_size),
        int((mm[2] - z_off) / s_thick)
    ]


def create_contour(intersections_list_mm, plane, x_size, y_size, z_size, pixel_size, x_offset, y_offset, z_offset,
                   slice_thickness):
    bitmap = None
    if plane == 'Sagittal':
        bitmap = np.zeros((z_size, y_size))
    elif plane == 'Coronal':
        bitmap = np.zeros((z_size, x_size))

    stale_variable = None
    for intersection in intersections_list_mm:
        for i in range(1, len(intersection), 2):
            point_0 = to_ind(intersection[i - 1], pixel_size, x_offset, y_offset, z_offset, slice_thickness)
            point_1 = to_ind(intersection[i], pixel_size, x_offset, y_offset, z_offset, slice_thickness)
            y_ = point_0[2]
            x_0, x_1 = None, None
            if plane == 'Sagittal':
                stale_variable = intersection[0][0]
                x_0 = point_0[1]
                x_1 = point_1[1]
            elif plane == 'Coronal':
                stale_variable = intersection[0][1]
                x_0 = point_0[0]
                x_1 = point_1[0]
            bitmap[y_, x_0:x_1] = 1

    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(bitmap, levels=0)

    contours = []
    for p in cp.collections[0].get_paths():
        v = np.array(p.vertices)
        y = v[:, 1] * slice_thickness + z_offset
        x = None
        contour = None
        if plane == 'Sagittal':
            x = v[:, 0] * pixel_size + y_offset
            zipped_xy = [(x[i], y[i]) for i in range(len(x))]
            contour = [[stale_variable, real_y, real_z] for real_y, real_z in zipped_xy]
        elif plane == 'Coronal':
            x = v[:, 0] * pixel_size + x_offset
            zipped_xy = [(x[i], y[i]) for i in range(len(x))]
            contour = [[real_x, stale_variable, real_z] for real_x, real_z in zipped_xy]
        contours.append(contour)

    return contours
