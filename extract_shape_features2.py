#
# I wrote the python code for all of the shape features in our repo except for smoothness
#
#
import numpy as np
from convert_mask_to_bounds import convert_mask_to_bounds, convert_contour_to_bound
import cv2
import scipy as scp
from scipy.spatial import distance
from PIL import Image, ImageDraw

## this file was made cause turns out everything is already done by default in opencv

# The bounds here are not referring to the human readable version. This is the raw bounds
# that comes out of opencv's findContours function. This is a singular bounds from the contours
# that opencv returns. So for example:
#
# im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# bounds = contours[0]
#
def get_morph_features(bounds):
    human_readable_bounds = convert_contour_to_bound(bounds)
    xy = order(human_readable_bounds)
    (xc,yc,area) = get_centroid(bounds)
    dist = []
    for each in bounds:
        dist.append(distance.euclidean(each,[xc,yc]))
    dist = np.array(dist)
    dist_min=np.min(dist)
    dist_max=np.max(dist)

    # Variance and co-variance in distance
    stdv = np.std(dist)
    cov = np.power(stdv,2)

    # Maximum Area and Area Ratio
    max_area = np.pi*np.power(dist_max,2)
    Area_Ratio = area/max_area

    # Ratio between average distance and maximum distance
    dist_mean = np.mean(dist)
    dist_ratio = dist_mean/dist_max

    # normalizing distance to find the variance and std
    dists_std = np.std(dist/dist_max)
    dists_cov = np.power(dists_std,2)

    # new distance ratio defined
    dratio = distratio(xy)

    # area to perimeter ratio
    perimeter = cv2.arcLength(bounds,True)
    paratio= np.power(perimeter,2)/area

    # Fourier Descriptors of boundary
    fft_vals = np.real(np.fft.fft(bounds[:,0][:,0] + 1j*bounds[:,0][:,1]))
    if fft_vals[1:].shape[0] < 10:
        fft_vals = np.hstack([fft_vals,np.full(10 - fft_vals[1:].shape[0], np.nan)])
    fd = fft_vals[1:]
    
    # Fractal Dimension and Invariant Moment
    frac_dim, huMoment = get_fractal_dimension(bounds)
    huMoment = np.transpose(huMoment)
    # eventually calculate smoothness features
    smoothness = 0

    return (Area_Ratio, dratio, dists_std, dists_cov, dratio, paratio, smoothness, huMoment, frac_dim, fd)


def get_fractal_dimension(cnt):
    # Transform contour into a binary mask
    a = cnt[:,0][:,0] - np.min(cnt[:,0][:,0])
    b = cnt[:,0][:,1] - np.min(cnt[:,0][:,1])
    Z = np.zeros([max(a)+1,max(b)+1])
    Z[a,b] = 1
    
    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.divide(1,counts), 1)

    # Invariant Moment Calculation
    moments = cv2.moments(Z)
    huMoments = cv2.HuMoments(moments)

    return (coeffs[0], huMoments)

def boxcount(Z, k):
    S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),np.arange(0, Z.shape[1], k), axis=1)
    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k*k))[0])



# Just pythonified patrick's code
def distratio(xy):
    n, m= xy.shape
    points = np.round(np.linspace(1,n,max(n*.01,3))).astype(int) - 1
    lx = xy[points,0]
    ly = xy[points,1]
    lxy = np.array([lx,ly])
    n,m = lxy.shape

    dislong = np.zeros([m-1,1])
    for i in range(0,m-1):
        dislong[i] = np.sqrt(np.power(lx[i]-lx[i+1],2) + np.power(ly[i]-ly[i+1],2))

    m,n = xy.shape
    dis_short = np.zeros([m-1,1])
    for i in range(0, m-1):
        #    dis_short(b)=sqrt((xy(b,1)-xy(b+1,1)).^2+(xy(b,2)-xy(b+1,2)).^2);
        part_a = np.power(xy[i,0]-xy[i+1,0],2)
        part_b = np.power(xy[i,1]-xy[i+1,1], 2)
        dis_short[i] = np.sqrt(part_a + part_b)
    
    dl=np.sum(dislong)
    ds=np.sum(dis_short)
    dratio=dl/ds
    return dratio

def get_centroid(cnt):
    M = cv2.moments(cnt)
    xc = (M["m10"] / M["m00"])
    yc = (M["m01"] / M["m00"])
    area = cv2.contourArea(cnt)
    return (xc,yc,area)

def start_bounds():
    a = np.array([[[1684, 1176]],

       [[1684, 1177]],

       [[1684, 1178]],

       [[1684, 1179]],

       [[1683, 1180]],

       [[1682, 1180]],

       [[1682, 1181]],

       [[1682, 1182]],

       [[1682, 1183]],

       [[1683, 1183]],

       [[1684, 1184]],

       [[1684, 1185]],

       [[1685, 1185]],

       [[1685, 1184]],

       [[1685, 1183]],

       [[1685, 1182]],

       [[1685, 1181]],

       [[1685, 1180]],

       [[1685, 1179]],

       [[1685, 1178]],

       [[1685, 1177]],

       [[1685, 1176]]], dtype=np.int32)

    a = np.flip(a, axis=0)
    
    return a

def order(c):
    ci = np.vstack((c['x'],c['y']))
    n,m = ci.shape
    xy = np.zeros([m,n])
    for i in reversed(range(0,m)):
        xy[m-1-i,0] = ci[0,i]
        xy[m-1-i,1] = ci[1,i]
    return xy

bounds = start_bounds()
Area_Ratio, dratio, dists_std, dists_cov, dratio, paratio, smoothness, huMoment, frac_dim, fd = get_morph_features(bounds)
print("Area_Ratio: ", Area_Ratio)
print("dratio: ", dratio)
print("dist_std:",dists_std)
print("dist_var:",dists_cov)
print("dratio:",dratio)
print("paratio:",paratio)
print("smoothness:",smoothness)
print("humoment:",huMoment)
print("frac_dm:",frac_dim)
print("fd:",fd)