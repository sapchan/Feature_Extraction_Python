import numpy as np
import scipy as scp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import cv2
#from skimage.feature import greycomatrix, greycoprops


class Extract_CGT_Features():
        
        def __init__(self, rgb_image, mask_image, threshold):
                self.rgb = rgb_image
                self.mask = mask_image
                ret, contours = cv2.findContours(self.mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                contours = np.array(contours)
                contour_locations = self.get_valid_contours(contours,threshold)
                self.contours = contours[contour_locations]
                self.centroids = self.get_centroids()
                self.create_probability_matrix_for_subgraphs(self.centroids)

        def get_valid_contours(self, contours, threshold):
                areas = np.array([])
                for idx, cnt in enumerate(contours):
                        area = cv2.contourArea(cnt)
                        areas = np.append(areas,area)
                valid_contours = np.where(areas > threshold)[0]
                return valid_contours
        
        def get_centroids(self):
                centroids = np.array([0,0])
                for idx,cnt in enumerate(self.contours):
                        M = cv2.moments(cnt)
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        coords = np.array([cx,cy])
                        centroids = np.vstack([centroids,coords])
                centroids = centroids[1:]
                return centroids

        def create_probability_matrix_for_subgraphs(self, centroids):
                D = pd.DataFrame(euclidean_distances(centroids))
                P = D.applymap(self.probability_computation)
                P = P.replace([np.inf, -np.inf], 0)
                E = P.applymap(self.should_edge_be_created)
                pos = {}
                node_angle = {}
                i_vert = np.where(np.triu(E) == True)[0]
                j_vert = np.where(np.triu(E) == True)[1]
                e = np.array([0,0])
                for idx,vertex in enumerate(i_vert):
                        i = i_vert[idx]
                        j = j_vert[idx]
                        edge_relation = np.array([i,j])
                        e = np.vstack([e,edge_relation])

                e = e[1:]
                v = np.linspace(0,len(centroids)-1, len(centroids)).astype(int)
                
                G = nx.Graph()
                G.add_nodes_from(v)
                G.add_edges_from(e)

                for idx,node in enumerate(v):
                        pos[idx] = self.centroids[idx]
                        (x,y),(MA,ma),angle = cv2.fitEllipse(self.contours[idx])
                        node_angle[idx] = angle

                # Create an empty co-occuring matrix
                bins = [i for i in range(0,190,10)]
                bins = pd.DataFrame(columns=bins,index=bins)
                bins = bins.fillna(0)

                # This gets us all of our subgraphs that are strongly connected based off of Tarjan's algorithm
                groups = nx.strongly_connected_component_subgraphs(G.to_directed())
                num_groups = nx.number_strongly_connected_components(G.to_directed())

                # Now we iterate through all of these connected subgraphs
                norm_co_mats = {}
                for idx,comp in enumerate(groups):
                        comp = comp.to_undirected()
                        angles = np.around([node_angle.get(key) for key in [i for i in comp.node]], decimals=-1).astype(int)
                        co_mat = self.create_cooccurence_matrix(angles)
                        norm_co_mat = np.divide(co_mat,np.sum(np.sum(co_mat)))
                        norm_co_mats[idx] = norm_co_mat
                print(norm_co_mats)
                


        def probability_computation(self, x):
                a = .5
                x = np.power(x,-a)
                return x

        def should_edge_be_created(self, probability):
                threshold = .07
                return probability > threshold


        # a helper function to help us create the co-occurence matrix when given all the angles
        def create_cooccurence_matrix(self,angles):
                a = np.zeros([19,1])
                for i in angles:
                        a[int(i/10)] = a[int(i/10)] + 1
                
                data=np.triu(np.dot(a,a.transpose()))
                for j in range(0,19):
                        data[j,j] = np.sqrt(data[j,j])
                df = pd.DataFrame(data=data)
                return df