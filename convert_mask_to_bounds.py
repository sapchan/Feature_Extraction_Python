import numpy as np
import cv2

def convert_mask_to_bounds(mask):
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    bounds = {}
    for idx,each in enumerate(contours):
        x = np.array([])
        y = np.array([])
        for idx2,points in enumerate(each):
            x = np.append(x,points[0][0])
            y = np.append(y,points[0][1])
        bounds[idx] = {'x':x,'y':y}
    return bounds

def convert_contour_to_bound(contour):
    bounds = {}
    x = np.array([])
    y = np.array([])
    for idx2,points in enumerate(contour):
        x = np.append(x,points[0][0])
        y = np.append(y,points[0][1])
    bounds = {'x':x,'y':y}
    return bounds 