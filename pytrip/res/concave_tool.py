#
#    Copyright (C) 2010-2021 PyTRiP98 Developers.
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
This file contains methods that create from a list of intersections (list of points).
Algorithm steps:
    1. Divide passed list of intersections into separate groups, that each represents a valid contour.
    2. For each group create a contour.
        2.1. Divide group of points into parts of contour, polygon representing each part imitates plot of a function
        2.2. Connect parts of contour (taking into account direction of connection)
    3. Return created contours.
Basic heuristics are used:
    1. searching for minimal distance between a part of contour and a point and ensuring that relation is symmetrical
        (if point A is the closest one to part B, check if part B is the closest one to point A)
    2. limiting distance between parts and points based on previous distances
        (next point cannot be too far away from parts)
"""

import math


class ListEntry:
    """Special type to hold data and other useful information"""
    def __init__(self):
        self.data = []
        self.last_distance = float('inf')


class SpecialPoint:
    """Special type to hold data and other useful information"""
    def __init__(self, point):
        self.point = point
        self.is_appended = False


# -------------------- start block of utility methods --------------------
magic_max_distance_multiplier = 2


def map_points_to_special_points(points):
    return [SpecialPoint(p) for p in points]


def distance(p, q):
    """Return value of euclidean distance between two points in euclidean 3D space"""
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2)


def search_for_closest_part(point, parts):
    cp = None
    closest_distance = float('inf')
    for part in parts:
        d = distance(point.point, part.data[-1])
        if d < closest_distance and d < magic_max_distance_multiplier * part.last_distance:
            closest_distance = d
            cp = part
    return cp


def search_for_closest_point(part, points):
    cp = None
    closest_distance = float('inf')
    for point in points:
        d = distance(point.point, part.data[-1])
        # distance should be the shortest one and should not be longer than *magic number* times last known distance
        if d < closest_distance and d < magic_max_distance_multiplier * part.last_distance:
            closest_distance = d
            cp = point
    return cp


def calculate_average_distance(parts_of_contour):
    average_distance = 3 * 2.0  # three times slice distance, but i dont know why - magic number
    if parts_of_contour:
        average_distance = math.fsum(part.last_distance for part in parts_of_contour) / len(parts_of_contour)
    return average_distance


# -------------------- end block of utility methods --------------------

# -------------------- start block of logic methods --------------------


def append_points_to_parts(current_points, parts_of_contour):
    """
    Appends points to parts of contour.
    For each part searches for the closest point, checks if relation is symmetrical and then appends point to the part.
    Can left unappended points if they are too far from current parts.
    Modifies passed list of parts of contour.
    """
    for current_part in parts_of_contour:
        closest_point = search_for_closest_point(current_part, current_points)
        # if that point exists - check if current part is the closest one to that point
        if closest_point:
            closest_part = search_for_closest_part(closest_point, parts_of_contour)
            # if the closest part to chosen point is the same as current part,
            #   then we can append point to part and mark the point as appended
            if closest_part == current_part and not closest_point.is_appended:
                current_part.last_distance = distance(closest_point.point, current_part.data[-1])
                current_part.data.append(closest_point.point)
                closest_point.is_appended = True


def create_parts_for_loose_points(current_points, parts_of_contour):
    """
    Creates new parts of contour for each point that is not appended yet.
    Modifies passed lists of parts of contour.
    """
    # instead of totally arbitrary initial value of last distance, calculate average
    # magic number is wrong, average is based on current real data
    # setting it to infinity causes bugs like in situation:
    #   There can be single-point-contour-part created, because that point is too far from other contours.
    #   The contour will be initialized with infinite last_distance value, which affects search_for_closest_* methods.
    #   It will remain single point for some loop steps, because other contours are closer to appended points.
    #   Then, second single-point-contour-part should be created, because another point is too far, but it won't!
    #       Why? Because of infinite last_distance from the first single point contour, that another point
    #       could be appended to the first single-point-contour-part, creating straight line between very distant points
    #       that should not be connected!
    average_distance = calculate_average_distance(parts_of_contour)
    for point in current_points:
        if not point.is_appended:
            # create single-point-part in parts
            new_list_entry = ListEntry()
            new_list_entry.data.append(point.point)
            new_list_entry.last_distance = average_distance
            point.is_appended = True
            parts_of_contour.append(new_list_entry)


def connect_closest_parts(parts):
    """
    Searches for two closest parts and connects them.
    Removes one of found parts from passed array.
    """
    a_reverse, b_reverse, part_a_to_merge, part_b_to_merge = search_for_closest_parts(parts)

    connect_two_parts(a_reverse, b_reverse, part_a_to_merge, part_b_to_merge)

    # remove used part from collection
    parts.remove(part_b_to_merge)


def search_for_closest_parts(parts):
    """
    Searches for two closest parts comparing distances from both ends of each part.
    """
    # placeholder for parts that should be merged
    part_a_to_merge = None
    part_b_to_merge = None
    # flags that tell how parts should be connected
    #   True - connect starting from the end of array
    #   False - connect starting from the start of array
    a_reverse = False
    b_reverse = False

    closest_distance = float('inf')
    part_b = None
    # look for two closest parts
    for part_a in parts:
        closest_to_part_b = None
        reverse_to_part_b = False
        # create collection without part_a, because it will be the closest one to itself
        parts_without_a = [p for p in parts if p != part_a]
        # same flag as b_reverse, but local one
        reverse_b = False
        # search for closest part to part_a, from both sides - aka part_b
        for last_point_a, reverse_a in [(part_a.data[-1], True), (part_a.data[0], False)]:
            for part_wo_a in parts_without_a:
                d = distance(last_point_a, part_wo_a.data[0])
                if d < closest_distance:
                    closest_distance = d
                    part_b = part_wo_a
                    reverse_b = False
                d = distance(last_point_a, part_wo_a.data[-1])
                if d < closest_distance:
                    closest_distance = d
                    part_b = part_wo_a
                    reverse_b = True

                if reverse_b:
                    last_point_b = part_b.data[-1]
                else:
                    last_point_b = part_b.data[0]

                # search for closest part to part_b, to make sure that relation is symmetrical
                closest_distance = float('inf')
                parts_without_b = [p for p in parts if p != part_b]
                for part_wo_b in parts_without_b:
                    d = distance(last_point_b, part_wo_b.data[-1])
                    if d < closest_distance:
                        closest_distance = d
                        closest_to_part_b = part_wo_b
                        reverse_to_part_b = True
                    d = distance(last_point_b, part_wo_b.data[0])
                    if d < closest_distance:
                        closest_distance = d
                        closest_to_part_b = part_wo_b
                        reverse_to_part_b = False
            # if the closest part to part_b is the as part_a considering proper end of array
            if closest_to_part_b == part_a and reverse_to_part_b == reverse_a:
                # save new closest parts and which ends are used
                a_reverse = reverse_a
                b_reverse = reverse_b
                part_a_to_merge = part_a
                part_b_to_merge = part_b

    return a_reverse, b_reverse, part_a_to_merge, part_b_to_merge


def connect_two_parts(a_reverse, b_reverse, part_a_to_merge, part_b_to_merge):
    """
    Connects two parts of contour depending on which ends of parts should be used.
    """
    #   True - connect starting from the end of array
    #   False - connect starting from the start of array
    if not a_reverse and not b_reverse:
        # adding two parts with their starts, must reverse part_a and then append part_b
        part_a_to_merge.data = list(reversed(part_a_to_merge.data))
        part_a_to_merge.data.extend(part_b_to_merge.data)
    elif not a_reverse and b_reverse:
        # adding to a start b end, so we can add a to b, and then assign b to a
        part_b_to_merge.data.extend(part_a_to_merge.data)
        part_a_to_merge.data = part_b_to_merge.data
    elif a_reverse and not b_reverse:
        # adding to a end b start, we just need to extend a
        part_a_to_merge.data.extend(part_b_to_merge.data)
    elif a_reverse and b_reverse:
        # adding to a end b end, we need to reverse b
        part_a_to_merge.data.extend(reversed(part_b_to_merge.data))


def group_points_for_multiple_contours(points_lists):
    """
    Divides points into groups that will be used to construct contour for each one.
    """
    # empty array for each group/contour
    multiple_contours = []
    # empty array for current group/contour
    current_contour = []

    for points_list in points_lists:
        # if current list of points (intersection) is not empty, append that list to current group
        if len(points_list) > 0:
            current_contour.append(points_list)
        # if it is empty it means that current contour has ended
        else:
            # save current contour as one of multiple ones
            multiple_contours.append(current_contour)
            # create new empty group/contour
            current_contour = []
    # save the last contour if it exists
    if current_contour:
        multiple_contours.append(current_contour)
    # remove empty ones
    #   it could be done earlier by filtering list passed as parameter
    multiple_contours = [contour for contour in multiple_contours if contour]
    # return only non-empty groups/contours
    return multiple_contours


# -------------------- end block of logic methods --------------------


# -------------------- start block of wrapper/helper methods --------------------
def create_contour_parts(points_lists):
    mapped_points = list(map(map_points_to_special_points, points_lists))

    parts_of_contour = []

    for current_points in mapped_points:
        append_points_to_parts(current_points, parts_of_contour)
        create_parts_for_loose_points(current_points, parts_of_contour)

    return parts_of_contour


def connect_all_parts(parts):
    guardian_counter = 0
    while len(parts) > 1:
        if guardian_counter > 10**6:
            raise RuntimeWarning("Connecting part in method connect_all_parts in concave_tool exceeded iteration limit")
        guardian_counter += 1

        connect_closest_parts(parts)

    # there is only one part left, that is whole contour
    contour = parts[0]
    # close contour by adding first element at the end of if
    contour.data.append(contour.data[0])

    # return list of points that make this contour
    return contour.data


def create_contour(points_lists):
    """
    Returns list of closed contours represented as sequences of points.
    """
    # group points to ensure that multiple contours are not considered as one
    grouped_points = group_points_for_multiple_contours(points_lists)
    contours = []

    for group in grouped_points:
        parts = create_contour_parts(group)
        contour = connect_all_parts(parts)
        contours.append(contour)

    return contours


# -------------------- end block of wrapper methods --------------------
