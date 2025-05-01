import math
import random
import numpy as np
import cv2

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    if p1 is None or p2 is None: return float('inf')
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_circularity(contour):
    """Calculates the circularity of a contour: 4*pi*Area / (Perimeter^2)."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return 0
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    return circularity


def get_average_color_bgr(frame, contour):
    """Calculates the average BGR color within a contour (used for display)."""
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    if cv2.countNonZero(mask) == 0: return (0, 0, 0)
    mean_val = cv2.mean(frame, mask=mask)
    return tuple(int(c) for c in mean_val[:3]) # BGR tuple