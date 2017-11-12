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
Problem 1a: Softmax Regression \& the XOR Problem

@author - Alexander G. Ororbia II
'''

def calPK(X, W, b):
	
	N = len(X)
	fk = np.dot(X, W) + np.repeat(b, [N], axis=  0) # 1*N
	efk = np.exp(fk)
	pk = np.divide(efk, np.sum(efk, axis = 1)[:,None])
	
	return pk

def calJFunction(X, y, theta, reg):
	
	W = theta[0]
	b = theta[1]
	N = len(X)
	pk = calPK(X, W, b)
	pY = []
	for i in range(len(pk)):
		pY.append(pk[i,y[i]])
	pY = np.array(pY)
	j_function = -1/len(X)*np.sum(np.log(pY)) +\
		reg/2*np.sum(np.sum(np.square(W), axis = 1), axis = 0)
	
	return j_function

def computeNumGrad(X,y,theta,reg): # returns approximate nabla
	# WRITEME: write your code here to complete the routine
	# f = WX + b
	
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
	W = theta[0]
	b = theta[1]
	N = len(X)
	D = np.shape(W)[0]
	K = np.shape(W)[1]
	
	y_one_hot = np.zeros((N, K))
	for i,y_label in enumerate(y):
		y_one_hot[i,int(y_label)] += 1
	y_one_hot = y_one_hot.astype(int)
	
	pk = calPK(X, W, b)
	dW = np.dot(X.transpose(), 1/N*(pk - y_one_hot)) + reg*W
	db = np.array([np.sum(1/N*(pk - y_one_hot), axis = 0)])
		
	return (dW,db)

def computeCost(X,y,theta,reg):
	# WRITEME: write your code here to complete the routine
	return calJFunction(X, y, theta, reg)

def predict(X,theta):
	# WRITEME: write your code here to complete the routine
	W = theta[0]
	b = theta[1]
	
	pk = calPK(X, W, b)
	return pk

def updateTheta(theta, theta_grad, step_size):
	
	for ii, param in enumerate(theta):
		theta[ii] = param - theta_grad[ii]* step_size
		
	return theta

def main():

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
	
	#Train a Linear Classifier
	
	# initialize parameters randomly
	D = X.shape[1]
	K = np.amax(y) + 1
	
	# initialize parameters in such a way to play nicely with the gradient-check!
	W = 0.01 * np.random.randn(D,K)
	b = np.zeros((1,K)) + 1.0
	theta = [W,b]
	
	# some hyperparameters
	reg = 1e-3 # regularization strength
	
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
	
	# Re-initialize parameters for generic training
	W = 0.01 * np.random.randn(D,K)
	b = np.zeros((1,K))
	theta = [W,b]
	
	n_e = 1000
	check = 10 # every so many pass/epochs, print loss/error to terminal
	step_size = 1e-1
	reg = 0.0 # regularization strength
	
	# gradient descent loop
	num_examples = X.shape[0]
	for i in range(n_e):
		# WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
		loss = computeCost(X, y, theta, reg)
		theta_grad = computeGrad(X, y, theta, reg)
		theta = updateTheta(theta, theta_grad, step_size)
		if i % check == 0:
			print ("iteration %d: loss %f" % (i, loss))
	
		# perform a parameter update
		# WRITEME: write your update rule(s) here
	 
	# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
# 	sys.exit(0) 
	
	# evaluate training set accuracy
	scores = predict(X,theta)
	#scores = np.dot(X, W) + b
	predicted_class = np.argmax(scores, axis=1)
	print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))
	
main()

