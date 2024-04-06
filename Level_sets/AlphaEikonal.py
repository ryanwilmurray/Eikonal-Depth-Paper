# Generates level sets of the eikonal depth for any value of alpha for
# a mixture of Gaussian distributions

import numpy as np
import scipy as sp
from numpy import linalg as la
import math

import os

import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=16)   
plt.rc('ytick', labelsize=16) 
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

N=128
ratio=4
n=(N+1+2*ratio)**2

thickness=ratio/N

# Generate points when using the grid
x=np.linspace(-1-thickness,1+thickness,N+1+2*ratio)
y=np.linspace(-1-thickness,1+thickness,N+1+2*ratio)
points=np.empty((0,2))
for i in x:
    for j in y:
        points=np.append(points,[[i,j]],axis=0)

#Find points that are in the boundary region
in_boundary = 1*(points[:,0]<=-1)+1*(points[:,0]>=1)+1*(points[:,1]<=-1)+1*(points[:,1]>=1)
in_boundary = 1*(in_boundary>0)

# Needed when using the grid
nEffective=(N+1)**2

X=points[:,0]
Y=points[:,1]
F=np.zeros((n,1))
f=np.zeros((n,1))

sx1=0.3
sx2=0.1
rho=0.0

cov = np.array([[sx1**2, rho*sx1*sx2], [rho*sx1*sx2, sx2**2]])
covInv=la.inv(cov)
covDet=la.det(cov)
Mu=np.array([[0.0,0.0],])

for i in range(n):
    X=np.array([points[i,:],])
    f[i]=np.exp(-0.5*(X-Mu)@covInv@np.transpose(X-Mu))/(np.pi*2*np.sqrt(covDet))

sx1=0.2
sx2=0.1
rho=0.0

cov = np.array([[sx1**2, rho*sx1*sx2], [rho*sx1*sx2, sx2**2]])
covInv=la.inv(cov)
covDet=la.det(cov)
Mu=np.array([[0.3,0.5],])

alpha=0.5

for i in range(n):
    X=np.array([points[i,:],])
    f[i]=0.5*(f[i]+np.exp(-0.5*(X-Mu)@covInv@np.transpose(X-Mu))/(np.pi*2*np.sqrt(covDet)))
    f[i]=f[i]**alpha

for i in range(n):
    F[i]=1/f[i]
    
F[np.nonzero(in_boundary)]=0


from scipy.spatial.distance import pdist,squareform

from sklearn.neighbors import radius_neighbors_graph

# Cut-off for the neighborhood graph
radius=2*ratio/N
sigma=0.2

sparse_graph = radius_neighbors_graph(points, radius, mode='connectivity',include_self=True)

dist_mat = radius_neighbors_graph(points, radius, mode='distance',include_self=True)
wght_mat=dist_mat
wght_mat.data=np.exp(-dist_mat.data**2/(2*sigma**2*radius**2))/(np.pi*nEffective*radius**4*sigma**4)

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
       
    #Letting num_iter be small after some burn-in period.
    if k > 100:
        num_iter = 1
    #print(len(considered_set))
    new_max_solved = 1000.
    new_max_solved_index = -1

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


fig1, ax = plt.subplots()
CS=plt.tricontour(points[:,0],points[:,1], values,levels=8, linewidths=1.4,colors='black')
fmt = '%.2f'
ax.clabel(CS,CS.levels[1:4], inline=1, fontsize=12, fmt=fmt)
plt.axis('square')
plt.xlim([-0.6, 0.7])
plt.ylim([-0.2, 0.7])
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16)
plt.tight_layout()
plt.savefig('depthcontours_GaussiansAlpha'+str(alpha).replace('.','')+'.pdf', dpi=300)

