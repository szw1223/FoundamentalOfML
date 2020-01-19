# -*- coding: utf-8 -*-
"""
File:   hw01.py
Author:  Zhewei Song
Date: 10th Sept 2019
Desc:

"""

""" =======================  Import dependencies ========================== """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

plt.close('all')  # close any open plots
"""
===============================================================================
============================ Question 1 =======================================
===============================================================================
"""
""" ======================  Function definitions ========================== """


def plotData(x1: object, t1: object, x2: object = None, t2: object = None, x3: object = None, t3: object = None, legend: object = []) -> object:
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'r')  # plot training data
    if (x2 is not None):
        p2 = plt.plot(x2, t2, 'g')  # plot true value
    if (x3 is not None):
        p3 = plt.plot(x3, t3, 'r')  # plot training data

    # add title, legend and axes labels
    plt.ylabel('|y - t|')  # label x and y axes
    plt.xlabel('k')

    if (x2 is None):
        plt.legend((p1[0]), legend)
    if (x3 is None):
        plt.legend((p1[0], p2[0]), legend)
    else:
        plt.legend((p1[0], p2[0], p3[0]), legend)

def fitdataLS(x, t, M):
    '''fitdataLS(x,t,M): Fit a polynomial of order M to the data (x,t) using LS'''
    # This needs to be filled in
    X = np.array([x ** m for m in range(M + 1)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return w

def fitdataIRLS(x, t, M, k):
    '''fitdataIRLS(x,t,M,k): Fit a polynomial of order M to the data (x,t) using IRLS'''
    b = []
    l = len(x)
    w_0 = fitdataLS(x, t, M)                        # To assign w_0 value of fitdataLS function
    X = np.array([x ** m for m in range(M + 1)]).T
    Y_0 = X @ w_0.T
    for i in range(l):                              # To calculate B diagonl matrix
        if abs(t[i] - Y_0[i]) <= k:
            b.append(1)
        else:
            b.append(k / (t[i] - Y_0[i]))
    B = np.diag(b)
    w = np.linalg.inv(X.T @ B @ X) @ X.T @ B @ t    # To calculate a new value of w
    w.flatten()
    wGapMax = abs(max(float(np.max(w - w_0)), float(np.min(w - w_0))))

    while wGapMax > 0.001:                          # To set a threshold
        b = []                                      # To initialize b and w
        w_0 = w
        Y_0 = X @ w_0.T
        for i in range(l):
            if abs(t[i] - Y_0[i]) <= k:
                b.append(1)
            else:
                b.append(k / (t[i] - Y_0[i]))
        B = np.diag(b)
        w = np.linalg.inv(X.T @ B @ X) @ X.T @ B @ t
        w.flatten()
        wGapMax = abs(max(float(np.max(w - w_0)), float(np.min(w - w_0))))
    return w


""" ======================  Variable Declaration ========================== """
M = 10                       # regression model order
k = .1                      # Huber M-estimator tuning parameter

""" =======================  Load Training Data ======================= """
data_uniform = np.load('TrainData.npy')
x1 = data_uniform[:, 0]
t1 = data_uniform[:, 1]

""" ========================  Train the Model ============================= """
wLS = fitdataLS(x1, t1, M)
wIRLS = fitdataIRLS(x1, t1, M, k)

""" ======================== Load Test Data  and Test the Model =========================== """
data_uniform = np.load('TestData.npy')
x2 = data_uniform[:, 0]
t2 = data_uniform[:, 1]
wLST = fitdataLS(x2, t2, M)
wIRLST = fitdataIRLS(x2, t2, M, k)
"""This is where you should load the testing data set. You shoud NOT re-train the model   """

""" ========================  Plot Results ============================== """

""" This is where you should create the plots requested """
xrange = np.arange(-4.5, 4.495, 0.009)  # get equally spaced points in the xrange
X = np.array([xrange ** m for m in range(wLS.size)]).T

YT_IRLS = []
YT_LS = []

for i in range(10):
    a = (i + 1) * 0.1
    YT_IRLS.append(float(np.linalg.norm(X@fitdataIRLS(x2, t2, 10, a).T - t2)))

for i in range(10):
    YT_LS.append(float(np.linalg.norm(X@fitdataLS(x2, t2, 10) - t2)))
k = list(range(0, 10))
li = []
for i in k:
    li.append(i/10)
plotData(li, YT_LS, li, YT_IRLS, None, None, ['LS Function', 'IRLS Function'])
plt.show()


