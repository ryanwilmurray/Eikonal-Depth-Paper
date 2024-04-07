# Calculates eikonal depth of a uniform distribution on a manifold
# consisting of a 2D sphere and a one-dimensional segment through it
import numpy as np
import scipy as sp
from numpy import linalg as la
from matplotlib.colors import ListedColormap
import math

import os

import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=16)   
plt.rc('ytick', labelsize=16) 
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

N=32
ratio=4

n_sphere=2200
n_line=300
n=n_sphere+n_line

x, y, z = np.random.multivariate_normal([0,0,0], [[1,0,0],[0,1,0],[0,0,1]], n_sphere).T
points_sphere=np.stack((x,y,z)).T
points_sphere /= la.norm(points_sphere, axis=1)[:, np.newaxis]

thickness=0.1
xline=np.zeros((n_line,))
yline=np.zeros((n_line,))
zline=-1-thickness+(2+2*thickness)*np.random.rand(n_line,)
points_line=np.stack((xline,yline,zline)).T

points=np.vstack((points_sphere,points_line))


#Find points that are in the boundary region
in_boundary = 1*(points[:,2]<=-1)+1*(points[:,2]>=1)
in_boundary = 1*(in_boundary>0)

nEffective=len(points)-len(np.nonzero(in_boundary))

F=np.ones((len(points),1))
F[np.nonzero(in_boundary)]=0
points_boundary=points[np.nonzero(in_boundary)]

#Create an array to color the different parts 
Col=np.ones((len(points),1))
Col[0:n_sphere-1]=0
Col[n_sphere:n]=1
Col[np.nonzero(in_boundary)]=2


from scipy.spatial.distance import pdist,squareform

from sklearn.neighbors import radius_neighbors_graph

# Cut-off for the neighborhood graph
radius=2*ratio/N
sigma=0.2

sparse_graph = radius_neighbors_graph(points, radius, mode='connectivity',include_self=True)

dist_mat = radius_neighbors_graph(points, radius, mode='distance',include_self=True)
wght_mat=dist_mat

sparseIinds=wght_mat.nonzero()[0]
sparseJinds=wght_mat.nonzero()[1]

for i in np.arange(len(sparseIinds)):
    I=sparseIinds[i]
    J=sparseJinds[i]
    if (I>=n_sphere) & (J>=n_sphere):
        # These are the points on the line
        wght_mat[I,J]=50*np.exp(-dist_mat[I,J]**2/(2*sigma**2*radius**2))/(np.sqrt(2*np.pi)*nEffective*radius**3*sigma**3)
    else:
        # These are the points on the sphere
        wght_mat[I,J]=50*np.exp(-dist_mat[I,J]**2/(2*sigma**2*radius**2))/(np.pi*nEffective*radius**4*sigma**4)

print('Minimum number of neighbors')
print(np.min(np.sum(1*(sparse_graph>0),axis=1)))
print('Maximum number of neighbors')
print(np.max(np.sum(1*(sparse_graph>0),axis=1)))


values = 100*np.ones(n)

#Initialize the solved set/list/values
solved_list = np.nonzero(in_boundary)[0]
solved_set = set(solved_list)
values[solved_list] = 0.

max_solved = 0.

from scipy.sparse import find

#Initialize the considered set
considered_set = set()
for i in solved_list:
    neighbors = set(find(sparse_graph[i,:])[1])
    considered_set = considered_set.union(neighbors.difference(solved_set))
update_set = considered_set

#Create a list so that neighbor_set_list[i] = {Neighbors of i}
neighbor_set_list = []
for i in range(n):
    neighbor_set_list.append(set(find(sparse_graph[i,:])[1]))
k = len(solved_set)


while considered_set:
    
    #This is just a progress indicator
    if k%100 == 0:
        print("{:.2f} %".format(k/n*100),end='\r')
    k += 1
    

    for i in update_set:
        neighbors = neighbor_set_list[i]
        solved_neighbors = list(neighbors.intersection(solved_set))  
        neighbor_weights = wght_mat[i,solved_neighbors].toarray()[0]
        neighbor_values = values[solved_neighbors]
            
            
        values[i] = (neighbor_values.dot(neighbor_weights) + np.sqrt(neighbor_values.dot(neighbor_weights)**2 - np.sum(neighbor_weights)*((neighbor_values**2).dot(neighbor_weights) - 1/F[i]**2)))/np.sum(neighbor_weights)
    
    #Here we use update set to track places in the considered set where values need to be updates (i.e. there is a new solved neighbor)
    l = list(considered_set)
    min_index = l[np.argmin(values[l])]
    solved_set.add(min_index)
    solved_list = list(solved_set)
    considered_set.remove(min_index)
    update_set = neighbor_set_list[min_index].difference(solved_set)
    considered_set = considered_set.union(update_set)


print("Calculations complete")


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
sctt=ax.scatter3D(points[:,0], points[:,1], points[:,2], c=values, cmap='hot',vmin=0,vmax=1)

fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
ax.set_box_aspect([1,1,1])
plt.savefig('depth_sphereline.jpg', dpi=300, bbox_inches='tight')
plt.close(fig)


IndicesToPlot=(1*(points[:,0]<=0)).nonzero()[0]
pointsToPlot=points[IndicesToPlot]
valuesToPlot=values[IndicesToPlot]

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
sctt=ax.scatter3D(pointsToPlot[:,0], pointsToPlot[:,1], pointsToPlot[:,2], c=valuesToPlot, cmap='hot',vmin=0,vmax=1)

fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
plt.xlim(-1,1)
ax.set_box_aspect([1,1,1])
plt.savefig('depth_sphereline_cut.jpg', dpi=300, bbox_inches='tight')
plt.close(fig)

