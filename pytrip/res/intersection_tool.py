def between(a, b, val):
    return a <= val < b or b <= val < a


def intersect(contour, plane, depth, i):
    x_0, y_0, z_0 = contour[i]
    x_1, y_1, _z_1 = contour[i+1]
    intersection = None
    if plane == 'Sagittal' and between(x_0, x_1, depth):
        a = (y_1 - y_0) / (x_1 - x_0)
        y = (depth - x_0) * a + y_0
        intersection = [depth, y, z_0]
    if plane == 'Coronal' and between(y_0, y_1, depth):
        a = (x_1 - x_0) / (y_1 - y_0)
        x = (depth - y_0) * a + x_0
        intersection = [x, depth, z_0]
    return intersection


def create_intersections(contour, plane, depth):
    intersections = []
    for i in range(-1, len(contour) - 1, 1):
        intersection = intersect(contour, plane, depth, i)
        if intersection:
            intersections.append(intersection)

    return intersections
