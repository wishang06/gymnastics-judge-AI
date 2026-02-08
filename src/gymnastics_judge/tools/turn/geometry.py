import numpy as np


def calculate_angle_3d(a, b, c):
    """Angle at b between ba and bc (0-180°)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def calculate_azimuth(shoulder_l, shoulder_r):
    """Azimuth of shoulder line in XY plane for rotation tracking (-180, 180]°."""
    dx = shoulder_r[0] - shoulder_l[0]
    dy = shoulder_r[1] - shoulder_l[1]
    return np.degrees(np.arctan2(dy, dx))
