import os 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import sys

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem 1b: Softmax Regression \& the Spiral Problem

@author - Guanjie Zheng
@author - Alexander G. Ororbia II (template provider)
'''		
	

def calPK(X, W, b):
	
	N = len(X)
	fk = np.dot(X, W) + np.repeat(b, [N], axis=  0) # N*k
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


def main(step_size, reg, f):

	np.random.seed(0)
	# Load in the data from disk
	path = os.getcwd() + '/data/spiral_train.dat'  
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
	
	#Train a Linear Classifier
	
	# initialize parameters randomly
	W = 0.1 * np.random.randn(D,K)
	b = np.zeros((1,K))
	theta = [W,b]
	
	# some hyperparameters
	n_e = 1000
	check = 10 # every so many pass/epochs, print loss/error to terminal
	
	# gradient descent loop
	for i in range(n_e):
		loss = computeCost(X, y, theta, reg)
		theta_grad = computeGrad(X, y, theta, reg)
		theta = updateTheta(theta, theta_grad, step_size)
		if i % check == 0:
			print ("iteration %d: loss %f" % (i, loss))
	
	 
	# TODO: remove this line below once you have correctly implemented/gradient-checked your various sub-routines
# 	sys.exit(0) 
	  
	# evaluate training set accuracy
	scores = predict(X,theta)
	#scores = np.dot(X, W) + b
	predicted_class = np.argmax(scores, axis=1)
	acc = np.mean(predicted_class == y)
	print ('training accuracy: %.2f' % acc)
	
	f.write("{0},{1},{2:.2f}\n".format(step_size, reg, acc))
	
	# plot the resulting classifier
	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
	Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
	Z = np.argmax(Z, axis=1)
	Z = Z.reshape(xx.shape)
	fig = plt.figure()
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	fig.savefig(os.path.join("out", "prob1b", 'spiral_linear_{0}_{1}.png'.format(step_size, reg)))
	
# 	plt.show()
	
f = open(os.path.join("out", "prob1b", "result"), "w")
f.write("step_size,reg,acc\n")
for step_size in [1e-3, 1e-2, 1e-1]:
	for reg in [0.001, 0.01, 0.1]:
		main(step_size, reg, f)
f.close()
