# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(12345)
np.set_printoptions(precision=2,suppress=True)


num_item = 5
num_month = 24
R_true = np.loadtxt('dataset1.txt',dtype=float)
R_true = np.reshape(R_true,[num_item,num_month])
R_1 = R_true
R_2 = R_true


p = 4  # window size
num_window = p * num_item  # of items in window
num_total = num_month * num_item

# form b
R_1 = R_1[:,p:]  # left cols are disposed
b = R_1.flatten(1)

# form A
R_2 = R_2[:,0:num_month-1]  # right cols are disposed
A = np.zeros([num_total-num_window, num_item + num_window])
for i in range (num_total-num_window):
	c_item  = i % num_item
	c_month = i / num_item
	A[i, c_item] = 1
	A[i, num_item:] = R_2[:,c_month:c_month+p].flatten(1)

# solve Ax = b
soln, resid, rank, s = np.linalg.lstsq(A,b)
mu = soln[0:num_item]
w = np.reshape(soln[num_item:],[num_item,p],order='F')


def estimate_month(m):
	if m < p: raise ValueError('Too early')
	if m < num_month:				
		return np.sum(R_true[:,m-p:m] * w) + mu		
	else:
		temp = R_true
		for i in range(num_month,m+1):
			cared = temp[:,i-p:i]
			est = np.sum(temp[:,i-p:i] * w) + mu
			est = np.reshape(est,[len(est),1])
			temp = np.hstack([temp,est])
		return est, temp



est, full = estimate_month(25)

# plt.figure()
# for i in range(num_item):
# 	plt.plot(full[i,:],'-o')	
# plt.show()









print 'All done'