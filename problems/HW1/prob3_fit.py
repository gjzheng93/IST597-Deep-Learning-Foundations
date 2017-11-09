import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib as mpl

'''
IST 597: Foundations of Deep Learning
Problem 3: Multivariate Regression & Classification

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
mpl.rcParams["figure.subplot.top"] = 0.8

# begin simulation

def sigmoid(z):
    
    return 1/(1+np.exp(-z))

def regress(X, theta):
    m = np.shape(X)[0]
    b = theta[0]
    w = theta[1]
    y_hat = sigmoid(np.repeat(b[None,:], m, axis = 0) + np.dot(X, w.transpose()))
    return y_hat

def bernoulli_log_likelihood(p, y):
    # WRITEME: write your code here to complete the routine
    return -1.0
    
def computeCost(X, y, theta, beta):
    m = np.shape(y)[0]
    y_hat = regress(X, theta)
    w = theta[1]
    loss = 1/(2*m)*np.sum(
        np.multiply(-y, np.log(y_hat)) - np.multiply(1-y, np.log(1-y_hat))) \
        + beta/(2*m)*np.sum(np.square(w))
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


def predict(X, theta):  
    return (regress(X, theta) > 0.5).astype(int)


def main(degree, beta, alpha, n_epoch, eps):
    path = os.getcwd() + '/data/prob3.dat'  
    data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
    
    positive = data2[data2['Accepted'].isin([1])]  
    negative = data2[data2['Accepted'].isin([0])]
     
    x1 = data2['Test 1']  
    x2 = data2['Test 2']
    
    # apply feature map to input features x1 and x2
    cnt = 0
    for i in range(1, degree+1):  
    	for j in range(0, i+1):
    		data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
    		cnt += 1
    
    data2.drop('Test 1', axis=1, inplace=True)  
    data2.drop('Test 2', axis=1, inplace=True)
    
    # set X and y
    cols = data2.shape[1]  
    X2 = data2.iloc[:,1:cols]  
    y2 = data2.iloc[:,0:1]
    
    # convert to numpy arrays and initalize the parameter array theta
    X2 = np.array(X2.values)  
    y2 = np.array(y2.values)  
    w = np.zeros((1,X2.shape[1]))
    b = np.array([0])
    theta2 = (b, w)
    
    L = computeCost(X2, y2, theta2, beta)
    print("-1 L = {0}".format(L))
    L_best = L
    i = 0
    halt = 0
    cost = []
    while(i < n_epoch and halt == 0):
        dL_db, dL_dw = computeGrad(X2, y2, theta2, beta)
        b = theta2[0]
        w = theta2[1]
        b = b - alpha * dL_db
        w = w - alpha * dL_dw
        theta2 = (b, w)
        L = computeCost(X2, y2, theta2, beta)
        
        print(" {0} L = {1}".format(i,L))
        i += 1
        
        
        if cost != []:
            if cost[-1]-L < eps:
                halt = 1
        cost.append(L)
        L_best = min(L_best, L)
        
    # print parameter values found after the search
    print("w = ",w)
    print("b = ",b)
    
    predictions = predict(X2, theta2)
    # compute error (100 - accuracy)
    err = 1 - np.sum(predictions == y2)/len(predictions)
    # WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
    print ('Error = {0}%'.format(err * 100.))
    
    
    # make contour plot
    xx, yy = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
    xx1 = xx.ravel()
    yy1 = yy.ravel()
    grid = np.c_[xx1, yy1]
    grid_nl = []
    # re-apply feature map to inputs x1 & x2
    for i in range(1, degree+1):  
    	for j in range(0, i+1):
    		feat = np.power(xx1, i-j) * np.power(yy1, j)
    		if (len(grid_nl) > 0):
    			grid_nl = np.c_[grid_nl, feat]
    		else:
    			grid_nl = feat
    probs = regress(grid_nl, theta2).reshape(xx.shape)
    
    fig = plt.figure(figsize = (20, 8))
    ax = fig.add_subplot(121)
    ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    
    ax.scatter(x1, x2, c=y2, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)
    
    ax.set(aspect="equal",
           xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
           xlabel="$X_1$", ylabel="$X_2$")
    
    ax2 = fig.add_subplot(122)
    
    ax2.plot([i+1 for i in range(len(cost))], cost, label = "loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    
    ax.text(0.4, 1.02, "alpha: {0}, n_epoch: {1}, beta: {2}\n loss: {3}, epoch: {4}\n Error = {5}%".\
                format(alpha, n_epoch, beta, round(L, 4), len(cost), round(err*100, 2)),
            transform=ax.transAxes,
            color='green', fontsize=30)
    
    fig.savefig(os.path.join("prob3", "alpha_{0}_n_epoch{1}_beta_{2}".format(alpha, n_epoch, beta).replace(".", "_") + ".png"))
#     plt.show()
    
# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p6_reg0' # will add a unique sub-string to output of this program
degree = 6 # p, degree of model (LEAVE THIS FIXED TO p = 6 FOR THIS PROBLEM)
beta = 100 # regularization coefficient
alpha = 0.01 # step size coefficient
n_epoch = 500000 # number of epochs (full passes through the dataset)
eps = 1e-8 # controls convergence criterion


for beta in [0, 0.1, 1, 10, 100]:
    for alpha in [0.001, 0.01, 0.1]:
        for n_epoch in [100000, 500000, 1000000]:
            main(degree, beta, alpha, n_epoch, eps)

