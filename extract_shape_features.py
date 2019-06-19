import numpy as np
from convert_mask_to_bounds import convert_mask_to_bounds, convert_contour_to_bound
import cv2
import scipy as scp




def getBounds(mask):
    bounds = convert_mask_to_bounds(mask)
    return bounds

def get_morph_features():
    c = start_bounds()
    xy = order(c)
    x = xy[:,0]
    y = xy[:,1]
    (xc,yc,area) = get_centroid(xy)
    distance = np.sqrt(np.power((x-xc),2)+np.power((y-yc),2))
    dist_min=np.min(distance)
    dist_max=np.max(distance)

    # Variance and co-variance in distance
    stdv = np.std(distance)
    cov = np.power(stdv,2)

    # Maximum Area and Area Ratio
    max_area = np.pi*np.power(dist_max,2)
    Area_Ratio = area/max_area

    # Ratio between average distance and maximum distance
    dist_mean = np.mean(distance)
    dist_ratio = dist_mean/dist_max

    # normalizing distance to find the variance and std
    dists = distance/dist_max
    dists_std = np.std(dists)
    dists_cov = np.power(dists_std,2)

    dratio = distratio(xy)
    


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
    print(dislong)

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


# this is built using Z.V. Kovarik's algorithm 1996
# adjusted for my purposes by Sacheth Jun,2019
def get_centroid(XY):
    # XY should be an nx2 method, if not throw an error
    n, m = XY.shape
    assert (m == 2), 'Ensure a valid boundsItem is selected'
    # cyclic shift of XY
    ST = np.roll(XY, n-1, axis=0)
    UV = ST+XY
    ST = ST-XY
    p_uv = np.prod(UV, axis=1)
    p_st = np.prod(ST, axis=1)
    CXY = np.matmul(3*p_uv+p_st,ST)/12
    # sum(UV(:,1).*ST(:,2)-UV(:,2).*ST(:,1))/4;
    part1 = np.multiply(UV[:,0], ST[:,1])
    part2 = np.multiply(UV[:,1],ST[:,0])
    part3 = part1-part2
    area = np.sum(part3)/4
    CXY = CXY/area
    CX = -CXY[0]
    CY=CXY[1]
    area = abs(area)
    return (CX,CY,area)

def start_bounds():
    boundsItem = {'x':[1684., 1684., 1683., 1682., 1682., 1683., 1684., 1684., 1685.,1685.],'y':[1176., 1179., 1180., 1180., 1183., 1183., 1184., 1185., 1185.,1176.]}
    return boundsItem

def order(c):
    ci = np.vstack((c['x'],c['y']))
    n,m = ci.shape
    xy = np.zeros([m,n])
    for i in reversed(range(0,m)):
        xy[m-1-i,0] = ci[0,i]
        xy[m-1-i,1] = ci[1,i]
    return xy


#    e
#boundsItem = start_bounds()
#get_centroid(boundsItem)e
#c = start_bounds()
#xy = order(c)
#print(get_centroid(xy))
#x = 

get_morph_features()