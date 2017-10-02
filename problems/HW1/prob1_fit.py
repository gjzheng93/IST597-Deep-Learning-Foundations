import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

'''
IST 597: Foundations of Deep Learning
Problem 1: Univariate Regression

@author - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.001 # step size coefficient
eps = 0.0001 # controls convergence criterion
n_epoch = 100 # number of epochs (full passes through the dataset)

# begin simulation

def regress(X, theta):
    m = np.shape(X)[0]
    b = theta[0]
    w = theta[1]
    y_hat = np.repeat(b[None,:], m, axis = 0) + np.dot(X, w.transpose())
    return y_hat

def gaussian_log_likelihood(mu, y):
	return -1.0
	
def computeCost(X, y, theta):
    m = np.shape(y)[0]
    y_hat = regress(X, theta)
    loss = 1/(2*m)*np.sum(np.square(y_hat - y))
    return loss
	
def computeGrad(X, y, theta): 
    m = y.shape[0]
    d = X.shape[1]
    diff = regress(X, theta) - y
    dL_db = 1/m*np.sum(diff, axis = 0)
    dL_dw = 1/m*np.sum(
        np.multiply(
            np.repeat(diff, d, axis = 1),
            X), axis = 0)
    dL_dw = np.array([dL_dw]).transpose() 
    nabla = (dL_db, dL_dw)
    return nabla

path = os.getcwd() + '/data/prob1.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 

# display some information about the dataset itself here
data.describe()

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)

plt.scatter(X, y)
plt.xlabel = "x"
plt.ylabel = "y"
plt.title = "food truch prediction"

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X, y, theta)
print("-1 L = {0}".format(L))
L_best = L
i = 0
cost = []
while(i < n_epoch):
    dL_db, dL_dw = computeGrad(X, y, theta)
    b = theta[0]
    w = theta[1]
    b = b - alpha * dL_db
    w = w - alpha * dL_dw
    theta = (b, w)
    L = computeCost(X, y, theta)
    print(" {0} L = {1}".format(i,L))
    i += 1
    
    cost.append(L)
    if abs(L-L_best) < eps:
        break
    L_best = min(L_best, L)
    
    
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

kludge = 0.25 # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)

plt.plot(X_test, regress(X_test, theta), label="Model")
plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)

# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch

plt.show() # convenience command to force plots to pop up on desktop
