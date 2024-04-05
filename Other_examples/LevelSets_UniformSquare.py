#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import scipy as sp
from numpy import linalg as la
import math

import os

from scipy.spatial.distance import pdist,squareform
from sklearn.neighbors import radius_neighbors_graph

import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=16)   
plt.rc('ytick', labelsize=16) 
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

N=128
ratio=4
#n=(N+1)**2
n=(N+1+2*ratio)**2

thickness=ratio/N

nEffective=(N+1)**2
x=np.linspace(-thickness,1+thickness,N+1+2*ratio)
y=np.linspace(-thickness,1+thickness,N+1+2*ratio)
points=np.empty((0,2))
for i in x:
    for j in y:
        points=np.append(points,[[i,j]],axis=0)

#Find points that are in the boundary region
in_boundary = 1*(points[:,0]<=0)+1*(points[:,0]>=1)+1*(points[:,1]<=0)+1*(points[:,1]>=1)
in_boundary = 1*(in_boundary>0)

X=points[:,0]
Y=points[:,1]
F=np.ones((n,1))
F[np.nonzero(in_boundary)]=1


# In[3]:


# Cut-off for the neighborhood graph
radius=ratio/N
sigma=0.2

sparse_graph = radius_neighbors_graph(points, radius, mode='connectivity',include_self=True)

dist_mat = radius_neighbors_graph(points, radius, mode='distance',include_self=True)
wght_mat=dist_mat
wght_mat.data=np.exp(-dist_mat.data**2/(2*sigma**2*radius**2))/(np.pi*nEffective*radius**4*sigma**4)

print('Minimum number of neighbors')
print(np.min(np.sum(1*(sparse_graph>0),axis=1)))
print('Maximum number of neighbors')
print(np.max(np.sum(1*(sparse_graph>0),axis=1)))


# In[4]:


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
    
    #Here we use update set to track places in the considered set where values need to be updates
    l = list(considered_set)
    min_index = l[np.argmin(values[l])]
    solved_set.add(min_index)
    solved_list = list(solved_set)
    considered_set.remove(min_index)
    update_set = neighbor_set_list[min_index].difference(solved_set)
    considered_set = considered_set.union(update_set)

print("Calculations complete")


# In[5]:


fig1, ax = plt.subplots()
CS=plt.tricontour(points[:,0],points[:,1], values,levels=10, linewidths=1.4,colors='black')
fmt = '%.2f'
ax.clabel(CS,CS.levels[1:7], inline=1, fontsize=12, fmt=fmt)
plt.axis('square')
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16)
plt.tight_layout()
plt.savefig('time_to_boundary_contours_square.pdf', dpi=300)
plt.close(fig1)
