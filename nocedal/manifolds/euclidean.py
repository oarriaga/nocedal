def transport(point_A, point_B, tangent_A):
    return tangent_A


def log(point_A, point_B):
    return point_B - point_A


def retract(point, tangent):
    return point + tangent


def scale(scalar, point):
    return scalar * point
