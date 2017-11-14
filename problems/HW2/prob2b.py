from __future__ import print_function
import os 
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from copy import deepcopy

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 2b: 2-Layer MLP for IRIS

@author - Guanjie Zheng
@author - Alexander G. Ororbia II (template provider)
'''


def calHpre(inp, W, b):

	N = len(inp)
	h_pre = np.dot(inp, W) + np.repeat(b, [N], axis=  0) # N*h
	return h_pre

def calH(inp, W, b):
	
	h_pre = calHpre(inp, W, b)
	h_func = np.maximum(h_pre, 0)
	
	return h_pre, h_func

def calPK(X, W1, b1, W2, b2, W3, b3):
	
	N = len(X)
	h1_pre, h1_func = calH(X, W1, b1)
	h2_pre, h2_func = calH(h1_func, W2, b2)
	fk = np.dot(h2_func, W3) + np.repeat(b3, [N], axis = 0) # N*k
	efk = np.exp(fk)
	pk = np.divide(efk, np.sum(efk, axis = 1)[:,None])
	
	return fk, pk

def calJFunction(X, y, theta, reg):
	
	W1 = theta[0]
	b1 = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	W3 = theta[4]
	b3 = theta[5]
	
	N = len(X)
	fk, pk = calPK(X, W1, b1, W2, b2, W3, b3)
	pY = []
	for i in range(len(pk)):
		pY.append(pk[i,y[i]])
	pY = np.array(pY)
	
	j_function = -1/len(X)*np.sum(np.log(pY)) \
		+ reg/2*np.sum(np.sum(np.square(W1), axis = 1), axis = 0) \
		+ reg/2*np.sum(np.sum(np.square(W2), axis = 1), axis = 0) \
		+ reg/2*np.sum(np.sum(np.square(W3), axis = 1), axis = 0)
	
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
	W3 = theta[4]
	b3 = theta[5]
	
	N = len(X)
	D = np.shape(X)[1]
	K = np.amax(y) + 1
	
	y_one_hot = np.zeros((N, K))
	for i,y_label in enumerate(y):
		y_one_hot[i,int(y_label)] += 1
	y_one_hot = y_one_hot.astype(int)
	
	h1_pre, h1_func = calH(X, W1, b1)
	h2_pre, h2_func = calH(h1_func, W2, b2)
	fk, pk = calPK(X, W1, b1, W2, b2, W3, b3)
	
	dJdfk = pk - y_one_hot
	dJdh2 = np.dot(1/N*(dJdfk), W3.transpose())  # N*K  K*D3
	dJdh2_pre = deepcopy(dJdh2)
	dJdh2_pre[np.where(h2_pre <= 0)] = 0
	dJdh1 = np.dot(dJdh2_pre, W2.transpose())    # N*D3  D3*D2
	dJdh1_pre = deepcopy(dJdh1)
	dJdh1_pre[np.where(h1_pre <= 0)] = 0
	
	dW3 = np.dot(h2_func.transpose(), 1/N*dJdfk) + reg*W3
	db3 = np.array([np.sum(1/N*dJdfk, axis = 0)])
	dW2 = np.dot(h1_func.transpose(), dJdh2_pre) + reg*W2
	db2 = np.array([np.sum(dJdh2_pre, axis = 0)])
	dW1 = np.dot(X.transpose(), dJdh1_pre) + reg*W1
	db1 = np.array([np.sum(dJdh1_pre, axis = 0)])
		
	return (dW1, db1, dW2, db2, dW3, db3)

def computeCost(X,y,theta,reg):
	return calJFunction(X, y, theta, reg)

def predict(X,theta):
	W1 = theta[0]
	b1 = theta[1]
	W2 = theta[2]
	b2 = theta[3]
	W3 = theta[4]
	b3 = theta[5]
	
	fk, pk = calPK(X, W1, b1, W2, b2, W3, b3)
	return pk

def updateTheta(theta, theta_grad, step_size):
	
	for ii, param in enumerate(theta):
		theta[ii] = param - theta_grad[ii]* step_size
		
	return theta

	
def create_mini_batch(X, y, start, end):
	# WRITEME: write your code here to complete the routine
	mb_x = X[start:end]
	mb_y = y[start:end]
	return (mb_x, mb_y)

		
def shuffle(X,y):
	ii = np.arange(X.shape[0])
	ii = np.random.shuffle(ii)
	X_rand = X[ii]
	y_rand = y[ii]
	X_rand = X_rand.reshape(X_rand.shape[1:])
	y_rand = y_rand.reshape(y_rand.shape[1:])
	return (X_rand,y_rand)

def main():
	
	np.random.seed(0)
	# Load in the data from disk
	path = os.getcwd() + '/data/iris_train.dat'  
	data = pd.read_csv(path, header=None) 
	
	# set X (training data) and y (target variable)
	cols = data.shape[1]  
	X = data.iloc[:,0:cols-1]  
	y = data.iloc[:,cols-1:cols] 
	
	# convert from data frames to numpy matrices
	X = np.array(X.values)  
	y = np.array(y.values)
	y = y.flatten()
	
	# load in validation-set
	path = os.getcwd() + '/data/iris_test.dat'
	data = pd.read_csv(path, header=None) 
	cols = data.shape[1]  
	X_v = data.iloc[:,0:cols-1]  
	y_v = data.iloc[:,cols-1:cols] 
	
	X_v = np.array(X_v.values)  
	y_v = np.array(y_v.values)
	y_v = y_v.flatten()
	
	
	# initialize parameters randomly
	D1 = X.shape[1]
	K = np.amax(y) + 1
	
	# initialize parameters randomly
	D2 = 100 # size of hidden layer
	D3 = 100 # size of hidden layer
	W1 = 0.01 * np.random.randn(D1,D2)
	b1 = np.zeros((1,D2))
	W2 = 0.01 * np.random.randn(D2,D3)
	b2 = np.zeros((1,D3))
	W3 = 0.01 * np.random.randn(D3,K)
	b3 = np.zeros((1,K))
	theta = [W1,b1,W2,b2,W3,b3]
	
	# some hyperparameters
	n_e = 10000
	n_b = 10
	step_size = 0.01 #1e-0
	reg = 1e-3 #1e-3 # regularization strength
	check = 10
	
	train_cost = []
	valid_cost = []
	
	min_loss_v = None
	patience = 0
	
	
# check gradient
	
# 	nabla_n = computeNumGrad(X,y,theta,reg)
# 	nabla = computeGrad(X,y,theta,reg)
# 	nabla_n = list(nabla_n)
# 	nabla = list(nabla)
# 	
# 	for jj in range(0,len(nabla)):
# 		is_incorrect = 0 # set to false
# 		grad = nabla[jj]
# 		grad_n = nabla_n[jj]
# 		err = np.linalg.norm(grad_n - grad) / (np.linalg.norm(grad_n + grad))
# 		if(err > 1e-5):
# 			print("Param {0} is WRONG, error = {1}".format(jj, err))
# 		else:
# 			print("Param {0} is CORRECT, error = {1}".format(jj, err))
	
	
	
	# gradient descent loop
	num_examples = X.shape[0]
	for i in range(n_e):
		X, y = shuffle(X,y) # re-shuffle the data at epoch start to avoid correlations across mini-batches
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
		#          you can use the "check" variable to decide when to calculate losses and record/print to screen (as in previous sub-problems)
	
		# WRITEME: write the inner training loop here (1 full pass, but via mini-batches instead of using the full batch to estimate the gradient)
		s = 0
		while (s < num_examples):
			# build mini-batch of samples
			X_mb, y_mb = create_mini_batch(X,y,s,min(s + n_b, num_examples))
			
			loss = computeCost(X_mb, y_mb, theta, reg)
			theta_grad = computeGrad(X_mb, y_mb, theta, reg)
			theta = updateTheta(theta, theta_grad, step_size)
			
			s += n_b
			
		if i % check == 0:
			print ("iteration %d: loss %f" % (i, loss))
			loss_t = computeCost(X, y, theta, reg)
			train_cost.append(loss_t)
			loss_v = computeCost(X_v, y_v, theta, reg)
			valid_cost.append(loss_v)
			
			print ("training loss: {0:.4f}", loss_t)
			print ("validation loss: {0:.4f}", loss_v)
			
			if min_loss_v == None:
				min_loss_v = loss_v
			else:
				min_loss_v = min(min_loss_v, loss_v)	
			
			if loss_v > min_loss_v:
				patience += 1
			else:
				patience = 0
			
			if patience > 10:
				print ("early stop")
				break
				
	
	print(' > Training loop completed!')
	# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
# 	sys.exit(0) 
	
	scores = predict(X,theta)
	predicted_class = np.argmax(scores, axis=1)
	print('training accuracy: {0}'.format(np.mean(predicted_class == y)))
	
	scores = predict(X_v,theta)
	predicted_class = np.argmax(scores, axis=1)
	print('validation accuracy: {0}'.format(np.mean(predicted_class == y_v)))
	
	# NOTE: write your plot generation code here (for example, using the "train_cost" and "valid_cost" list variables)
	
	
main()


