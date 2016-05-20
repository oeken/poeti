# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(12345)
np.set_printoptions(precision=2,suppress=True)
colors = ['#e47993','#7bb227','#4e0f91','#98332e','#5fabb2','#458186']

def symmetrize(A):
	temp = A + A.T
	temp[temp > 0] = 1
	return temp

def rand_range(h,l,s1,s2):
	return np.random.rand(s1,s2) * (h-l) + l

def kmeans(D, k):
	N = D.shape[0]
	dim = D.shape[1]
	ave = np.mean(D, axis=0)
	std = np.std(D, axis=0)
	centers = np.random.rand(k,dim) * std + mean		
	groups = np.random.randint(0,k,[N,1])
	for j in range(20):		
		for i in range(k):			
			centers[i,:] = np.mean(D[np.nonzero(groups == i)[0],:],axis=0)
		for i in range(N):
			centers - D[i,:]
			dist = np.sum(np.abs(centers-D[i,:])**2,axis=-1)**(1./2)
			groups[i] = np.argsort(dist)[0]		
	return groups, centers



buff = np.loadtxt('dataset.txt',dtype=float)
# buff = buff[:10]

size = len(buff)
N = size/2
data = np.zeros([N, 2])
for i in range(len(buff)):
	if i%2 == 0:
		data[i/2,0] = buff[i]
	else :
		data[i/2,1] = buff[i]


n = 5
neighbours = np.zeros([N,N])
distances = np.zeros([N,N])
for i in range(N):
	dist = np.sum(np.abs(data-data[i,:])**2,axis=-1)**(1./2)
	distances[i,:] = dist
	current_neighbours = np.argsort(dist)[1:1+n]	
	for j in current_neighbours:
		neighbours[i,j] = 1
neighbours = symmetrize(neighbours)


plt.figure()
plt.plot(data[:,0],data[:,1],'.')
edges = np.nonzero(neighbours == 1)
for i in range(len(edges[0])):		
	x1 = data[edges[0][i],:]
	x2 = data[edges[1][i],:]
	current = np.vstack([x1,x2])
	plt.plot(current[:,0],current[:,1],'-k',lw=0.5)
plt.axis('equal')


sigma = 0.3
weights = np.exp(-0.5 * (distances / sigma)**2) * neighbours
degrees = np.eye(N)
for i in range(N):
	degrees[i,i] = np.sum(weights[i,:])

laplacian = np.eye(N) - np.dot(weights, np.linalg.inv(degrees))

w, v = np.linalg.eig(laplacian)
index = np.argsort(w)


# groups = np.zeros([N,1])
# th = 0.025
# groups = v[:,index[1]] < th
# k = 2
# groups = kmeans(v[:,index[:k]],k)[0]

# plt.figure()
# for i in range(N):		
# 	plt.plot(data[i,0],data[i,1],'or', color=colors[int(groups[i,0])])
	

# for i in range(len(edges[0])):		
# 	x1 = data[edges[0][i],:]
# 	x2 = data[edges[1][i],:]
# 	current = np.vstack([x1,x2])
# 	plt.plot(current[:,0],current[:,1],'-k',lw=0.5)
# plt.axis('equal')

k = 3
groups_kmeans, centers = kmeans(data, k)

plt.figure()
for i in range(N):
	plt.plot(data[i,0],data[i,1],'o',color=colors[int(groups_kmeans[i,0])])
	
	

for i in range(len(edges[0])):		
	x1 = data[edges[0][i],:]
	x2 = data[edges[1][i],:]
	current = np.vstack([x1,x2])
	plt.plot(current[:,0],current[:,1],'-k',lw=0.5)

for i in range(k):
	plt.plot(centers[i,0],centers[i,0],'ob',lw=10)
plt.axis('equal')



plt.figure()
plt.plot(data[:,0],data[:,1],'.k')


plt.figure()
sns.distplot(v[:,index[1]], kde=False, rug=True);
plt.show(block=False)



