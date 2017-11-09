import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib as mpl

'''
IST 597: Foundations of Deep Learning
Problem 2: Polynomial Regression & 

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

mpl.rcParams["font.size"] = 18
mpl.rcParams["figure.subplot.left"] = 0.1
mpl.rcParams["figure.subplot.right"] = 0.95
mpl.rcParams["figure.subplot.bottom"] = 0.15
mpl.rcParams["figure.subplot.top"] = 0.82

# begin simulation

def regress(X, theta):
    m = np.shape(X)[0]
    b = theta[0]
    w = theta[1]
    y_hat = np.repeat(b[None,:], m, axis = 0) + np.dot(X, w.transpose())
    return y_hat

def gaussian_log_likelihood(mu, y):
    return -1.0
    
def computeCost(X, y, theta, beta):
    m = np.shape(y)[0]
    y_hat = regress(X, theta)
    w = theta[1]
    loss = 1/(2*m)*np.sum(np.square(y_hat - y)) + beta/(2*m)*np.sum(np.square(w))
    return loss
    
def computeGrad(X, y, theta, beta): 
    m = y.shape[0]
    d = X.shape[1]
    b = theta[0]
    w = theta[1]
    diff = regress(X, theta) - y
    dL_db = 1/m*np.sum(diff, axis = 0)
    dL_dw = 1/m*np.sum(
        np.multiply(
            np.repeat(diff, d, axis = 1),
            X), axis = 0)
    dL_dw = np.array([dL_dw])
    dL_dw = dL_dw + beta/m*w
    nabla = (dL_db, dL_dw)
    return nabla

def main(degree, beta, alpha, eps, n_epoch):

    path = os.getcwd() + '/data/prob2.dat'  
    data = pd.read_csv(path, header=None, names=['X', 'Y']) 
    
    # set X (training data) and y (target variable)
    cols = data.shape[1]  
    X = data.iloc[:,0:cols-1]  
    y = data.iloc[:,cols-1:cols] 
    
    # convert from data frames to numpy matrices
    X = np.array(X.values)
    y = np.array(y.values)
    
    # apply feature map to input features x1
    X_feature_map = X.copy()
    for i in range(2, degree+1):
        X_feature_map = np.concatenate((X_feature_map, np.power(X, i)), axis = 1)
    X = X_feature_map
    
    # convert to numpy arrays and initalize the parameter array theta 
    w = np.zeros((1,X.shape[1]))
    b = np.array([0])
    theta = (b, w)
    
    L = computeCost(X, y, theta, beta)
    print("-1 L = {0}".format(L))
    L_best = L
    i = 0
    cost = []
    while(i < n_epoch):
        dL_db, dL_dw = computeGrad(X, y, theta, beta)
        b = theta[0]
        w = theta[1]
        b = b - alpha * dL_db
        w = w - alpha * dL_dw
        theta = (b, w)
        L = computeCost(X, y, theta, beta)
        print(" {0} L = {1}".format(i,L))
        i += 1
    
        
        if cost != []:
            if cost[-1]-L < eps:
                break
        cost.append(L)
        L_best = min(L_best, L)
        
    # print parameter values found after the search
    print("w = ",w)
    print("b = ",b)
    
    kludge = 0.25
    # visualize the fit against the data
    X_test = np.linspace(data.X.min(), data.X.max(), 100)
    X_feat = np.expand_dims(X_test, axis=1)
    
    X_feat_map = X_feat.copy()
    for i in range(2, degree+1):
        X_feat_map = np.concatenate((X_feat_map, np.power(X_feat, i)), axis = 1)
    X_feat = X_feat_map
    
    fig = plt.figure(figsize = (20, 8))
    ax1 = fig.add_subplot(121)
    ax1.plot(X_test, regress(X_feat, theta), label="Model")
    ax1.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
    ax1.set_ylim((np.amin(y) - kludge, np.amax(y) + kludge))
    ax1.legend(loc="best")
    ax1.text(0.4, 1.1, "order of model: {0}  beta = {1}\n L = {2}, total epochs: {3}".format(degree,beta,round(L, 4),len(cost)),
            transform=ax1.transAxes,
            color='green', fontsize=30)
    
    ax2 = fig.add_subplot(122)
    ax2.plot([i+1 for i in range(len(cost))], cost, label = "loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    fig.savefig(os.path.join("prob2", "degree_" + str(degree) + "_beta_" + str(beta) +"_loss.png"))
#     plt.show()
    
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
degree = 15 # p, order of model
beta = 1 # regularization coefficient
alpha = 0.01 # step size coefficient
eps = 1e-8 # controls convergence criterion
n_epoch = 500000 # number of epochs (full passes through the dataset)


beta = 0
for degree in [1, 3, 7, 11, 15]:
    main(degree, beta, alpha, eps, n_epoch)

degree = 15
for beta in [0, 0.001, 0.01, 0.1, 1]:
    main(degree, beta, alpha, eps, n_epoch)


