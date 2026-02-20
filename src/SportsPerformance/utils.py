import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle between three 2D points.

    Args:
        a: First point (e.g., shoulder) as a list or tuple [x, y].
        b: Mid point (e.g., elbow) as a list or tuple [x, y].
        c: End point (e.g., wrist) as a list or tuple [x, y].

    Returns:
        The angle in degrees, between 0 and 180.
    """
    # Convert points to numpy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Create vectors from the points: vector from b to a, and vector from b to c
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Ensure angle is between 0 and 180
    if angle > 180.0:
        angle = 360 - angle

    return angle





#Old Implementation


# import numpy as np

# def calculate_angle(a, b, c):
#     """
#     Calculates the angle between three 2D points.

#     Args:
#         a: First point (e.g., shoulder) as a list or tuple [x, y].
#         b: Mid point (e.g., elbow) as a list or tuple [x, y].
#         c: End point (e.g., wrist) as a list or tuple [x, y].

#     Returns:
#         The angle in degrees.
#     """
#     # Convert points to numpy arrays
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     # Create vectors from the points
#     # Vector from b to a, and vector from b to c
#     ba = a - b
#     bc = c - b

#     # Calculate the dot product and the magnitudes of the vectors
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

#     # Use arccos to get the angle in radians, then convert to degrees
#     angle = np.degrees(np.arccos(cosine_angle))

#     # Ensure angle is between 0 and 180
#     if angle > 180.0:
#         angle = 360 - angle

#     return angle