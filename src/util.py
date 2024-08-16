import numpy as np
import math

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def calculate_distance(results):
    x1, y1 = results[0]
    x2, y2 = results[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance