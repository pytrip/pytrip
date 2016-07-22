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
import numpy as np
from math import pi, sin, cos, acos, asin


def get_basis_from_angles(gantry, couch):
    gantry /= 180.0 / pi
    couch /= -180.0 / pi
    a = -np.array([sin(gantry) * cos(couch), -cos(gantry), sin(couch) * sin(gantry)])
    c = -np.array([sin(gantry + pi / 2) * cos(couch), -cos(gantry + pi / 2), sin(couch) * sin(gantry + pi / 2)])
    b = np.cross(a, c)
    return [a, b, c]


def angles_from_trip(gantry, couch):
    gantry += 90
    couch = -(couch + 90)
    return gantry, couch


def angles_to_trip(gantry, couch):
    gantry -= 90
    couch = -couch - 90
    return gantry, couch


def vector_to_angles(vec):
    gantry = acos(vec[1])
    couch = asin(vec[2] / sin(gantry))
    return gantry / pi * 180, couch / pi * 180


def point_in_polygon(x, y, polygon):
    intersects = 0
    n = len(polygon)
    x1 = polygon[0][0]
    y1 = polygon[0][1]
    for i in range(n + 1):
        x2 = polygon[i % n][0]
        y2 = polygon[i % n][1]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= xinters:
                        intersects += 1
        x1 = x2
        y1 = y2
    if intersects % 2 != 0:
        return True
    else:
        return False


def get_nearest_point(point, contour):
    length = 100000000.0
    p_out = None
    p2 = np.array(point)
    for p in contour:
        temp_len = sum((p2 - np.array(p))**2)
        if temp_len < length:
            length = temp_len
            p_out = p
    return p_out


def max_list(a, b):
    return [max(x, y) for x, y in zip(a, b)]


def min_list(a, b):
    return [min(x, y) for x, y in zip(a, b)]


def array_to_point_array(points, offset):
    point = [[points[3 * i] - offset[0], points[3 * i + 1] - offset[1], points[3 * i + 2] - offset[2]]
             for i in range(len(points) / 3)]
    return point


def get_area_contour(polygon):
    points = np.zeros((len(polygon), 2))
    points[0:len(polygon)] = np.array(polygon)[:, 0:2]
    points[-1] = np.array(polygon[0])
    dx_dy = np.array([points[i + 1] - points[i] for i in range(len(points) - 1)])
    points = np.array([(points[i + 1] + points[i]) / 2 for i in range(len(points) - 1)])
    area = -sum(points[:, 1] * dx_dy[:, 0])
    return area


def get_x_intersection(y, polygon):
    intersections = []
    x1 = polygon[0][0]
    y1 = polygon[0][1]
    n = len(polygon)
    for i in range(n + 1):
        x2 = polygon[i % n][0]
        y2 = polygon[i % n][1]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                x = (x2 - x1) / (y2 - y1) * (y - y1) + x1
                intersections.append(x)
        x1 = x2
        y1 = y2
    return intersections


# find a short distance probably not the shortest
def short_distance_polygon_idx(poly1, poly2):
    d = 10000000
    n1 = len(poly1)
    n2 = len(poly2)
    i1 = 0
    i2 = 0
    for i in range(n2):
        d1 = (poly2[i][0] - poly1[i1][0])**2 + (poly2[i][1] - poly1[i1][1])**2
        if d1 < d:
            i2 = i
            d = d1
    for i in range(n1):
        d2 = (poly2[i2][0] - poly1[i][0])**2 + (poly2[i2][1] - poly1[i][1])**2
        if d2 < d:
            i1 = i
            d = d2
    for i in range(n2):
        d1 = (poly2[i][0] - poly1[i1][0])**2 + (poly2[i][1] - poly1[i1][1])**2
        if d1 < d:
            i2 = i
            d = d1
    for i in range(n1):
        d2 = (poly2[i2][0] - poly1[i][0])**2 + (poly2[i2][1] - poly1[i][1])**2
        if d2 < d:
            i1 = i
            d = d2

    return i1, i2, float(d)**0.5
