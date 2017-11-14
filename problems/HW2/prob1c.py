import os 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import sys
from copy import deepcopy

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 1c: MLPs \& the XOR Problem

@author - Guanjie Zheng
@author - Alexander G. Ororbia II (template provider)
'''


def calHpre(X, W1, b1):

	N = len(X)
	h_pre = np.dot(X, W1) + np.repeat(b1, [N], axis=  0) # N*h
	return h_pre

def calH(X, W1, b1):
	
	h_pre = calHpre(X, W1, b1)
	h_func = np.maximum(h_pre, 0)
	
	return h_func

def calPK(X, W1, b1, W2, b2):
	
	N = len(X)
	h_func = calH(X, W1, b1)
	
	fk = np.dot(h_func, W2) + np.repeat(b2, [N], axis = 0) # N*k
	efk = np.exp(fk)
	pk = np.divide(efk, np.sum(efk, axis = 1)[:,None])
	
	return pk

def calJFunction(X, y, theta, reg):
	
	W1 = theta[0]
	b1 = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	
	N = len(X)
	pk = calPK(X, W1, b1, W2, b2)
	pY = []
	for i in range(len(pk)):
		pY.append(pk[i,y[i]])
	pY = np.array(pY)
	
	j_function = -1/len(X)*np.sum(np.log(pY)) \
		+ reg/2*np.sum(np.sum(np.square(W1), axis = 1), axis = 0) \
		+ reg/2*np.sum(np.sum(np.square(W2), axis = 1), axis = 0)
	
	return j_function

def computeNumGrad(X,y,theta,reg): # returns approximate nabla
	# WRITEME: write your code here to complete the routine
	
	eps = 1e-5
	theta_grad =  deepcopy(theta)
	for ii, param_grad in enumerate(theta_grad):
		it = np.nditer(param_grad, flags = ["multi_index"])
		while not it.finished:
			value = it[0]
			index = it.multi_index
			theta_grad[ii][index] = 0
			it.iternext()
	# NOTE: you do not have to use any of the code here in your implementation...
	for ii, param in enumerate(theta):
		it = np.nditer(param, flags = ["multi_index"])
		while not it.finished:
			value = it[0]
			index = it.multi_index
			theta_add_eps = deepcopy(theta)
			theta_add_eps[ii][index] = theta[ii][index] + eps
			j_add_eps = calJFunction(X, y, theta_add_eps, reg)
			theta_minus_eps = deepcopy(theta)
			theta_minus_eps[ii][index] = theta[ii][index] - eps
			j_minus_eps = calJFunction(X, y, theta_minus_eps, reg)
			grad = (j_add_eps-j_minus_eps)/(2*eps)
			theta_grad[ii][index] = grad
			it.iternext()
	return theta_grad	
	
def computeGrad(X,y,theta,reg): # returns nabla
	# WRITEME: write your code here to complete the routine
	W1 = theta[0]
	b1 = theta[1]	
	W2 = theta[2]
	b2 = theta[3]
	
	N = len(X)
	D = np.shape(W1)[0]
	K = np.shape(W2)[1]
	
	y_one_hot = np.zeros((N, K))
	for i,y_label in enumerate(y):
		y_one_hot[i,int(y_label)] += 1
	y_one_hot = y_one_hot.astype(int)
	
	h_pre = calHpre(X, W1, b1)
	h_func = calH(X, W1, b1)
	pk = calPK(X, W1, b1, W2, b2)
	
	dJdfk = pk - y_one_hot
	dJdh = np.dot(1/N*(dJdfk), W2.transpose())
	dJdh_pre = deepcopy(dJdh)
	dJdh_pre[np.where(h_pre <= 0)] = 0
	
	dW1 = np.dot(X.transpose(), dJdh_pre) + reg*W1
	db1 = np.array([np.sum(dJdh_pre, axis = 0)])
	dW2 = np.dot(h_func.transpose(), 1/N*dJdfk) + reg*W2
	db2 = np.array([np.sum(1/N*dJdfk, axis = 0)])
		
	return (dW1, db1, dW2, db2)

def computeCost(X,y,theta,reg):
	# WRITEME: write your code here to complete the routine
	return calJFunction(X, y, theta, reg)

def predict(X,theta):
	# WRITEME: write your code here to complete the routine
	W1 = theta[0]
	b1 = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	
	pk = calPK(X, W1, b1, W2, b2)
	return pk

def updateTheta(theta, theta_grad, step_size):
	
	for ii, param in enumerate(theta):
		theta[ii] = param - theta_grad[ii]* step_size
		
	return theta

def main(step_size, reg, h, f):
	
	np.random.seed(0)
	# Load in the data from disk
	path = os.getcwd() + '/data/xor.dat'  
	data = pd.read_csv(path, header=None) 
	
	# set X (training data) and y (target variable)
	cols = data.shape[1]  
	X = data.iloc[:,0:cols-1]  
	y = data.iloc[:,cols-1:cols] 
	
	# convert from data frames to numpy matrices
	X = np.array(X.values)  
	y = np.array(y.values)
	y = y.flatten()
	
	# initialize parameters randomly
	D = X.shape[1]
	K = np.amax(y) + 1
	
	# initialize parameters in such a way to play nicely with the gradient-check! 
	W1 = 0.05 * np.random.randn(D,h) #0.01 * np.random.randn(D,h)
	b1 = np.zeros((1,h)) + 1.0
	W2 = 0.05 * np.random.randn(h,K) #0.01 * np.random.randn(h,K)
	b2 = np.zeros((1,K)) + 1.0
	theta = [W1,b1,W2,b2] 
	
	nabla_n = computeNumGrad(X,y,theta,reg)
	nabla = computeGrad(X,y,theta,reg)
	nabla_n = list(nabla_n)
	nabla = list(nabla)
	
	for jj in range(0,len(nabla)):
		is_incorrect = 0 # set to false
		grad = nabla[jj]
		grad_n = nabla_n[jj]
		err = np.linalg.norm(grad_n - grad) / (np.linalg.norm(grad_n + grad))
		if(err > 1e-8):
			print("Param {0} is WRONG, error = {1}".format(jj, err))
		else:
			print("Param {0} is CORRECT, error = {1}".format(jj, err))
	
	# re-init parameters
	 #100 # size of hidden layer
	W1 = 0.01 * np.random.randn(D,h)
	b1 = np.zeros((1,h))
	W2 = 0.01 * np.random.randn(h,K)
	b2 = np.zeros((1,K))
	theta = [W1,b1,W2,b2] 
	
	# some hyperparameters
	n_e = 10000
	check = 10 # every so many pass/epochs, print loss/error to terminal
	
	list_loss = []
	cnt_epoch = []
		
	# gradient descent loop
	for i in range(n_e):
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
		loss = computeCost(X, y, theta, reg)
		theta_grad = computeGrad(X, y, theta, reg)
		theta = updateTheta(theta, theta_grad, step_size)
		if i % check == 0:
			print ("iteration %d: loss %f" % (i, loss))
			cnt_epoch.append(i)
			list_loss.append(loss)
	 
	# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
# 	sys.exit(0) 
	
	scores = predict(X,theta)
	predicted_class = np.argmax(scores, axis=1)
	acc = np.mean(predicted_class == y)
	print ('training accuracy: %.2f' % acc)
	
	f.write("{0},{1},{2},{3:.2f}\n".format(step_size, reg, h, acc))
	
	fig = plt.figure()
	plt.plot(cnt_epoch, list_loss)
	plt.xlabel("epoch")
	plt.ylabel("training loss")
	plt.title("step size={0}, reg={1}, h={2}".format(step_size,reg,h))
	fig.savefig(os.path.join("out", "prob1c", 'loss_{0}_{1}_{2}.png'.format(step_size, reg, h)))

f = open(os.path.join("out", "prob1c", "result"), "w")
f.write("step_size,reg,h,acc\n")
for step_size in [1e-3, 1e-2, 1e-1]:
	for reg in [0.001, 0.01, 0.1]:
		for h in [10, 50, 100]:
			main(step_size, reg, h, f)
f.close()
	
