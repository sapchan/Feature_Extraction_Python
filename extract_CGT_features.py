import numpy as np
from convert_mask_to_bounds import convert_mask_to_bounds, convert_contour_to_bound
import cv2
import scipy as scp
from scipy.spatial import distance
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw

def extract_CGT_features(contours):
    # calculate principal axis
    axis = np.zeros([len(contours), 2])
    for idx,cnt in enumerate(contours):
        pca = PCA(n_components=2)
        pca.fit(np.flip(cnt[:,0], axis=0))
        major = pca.components_[0]
        axis[idx,:] = major