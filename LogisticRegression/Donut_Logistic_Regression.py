# -*- coding: utf-8 -*-
"""
Author: Jungho Park
Date: October 14 2018

"""
import numpy as np
import matplotlib.pyplot as plt

# Number of data points for inside and outside donuts
N = 400

# mean radius of inside and outside donuts
R_inside = 3
R_outside = 8

def sigmoid(x):
    # Sigmoid function
    # Sigmoid is 0.5 at x=0.5
    # Return: (0,1)
    return 1 / (1 + np.exp(-x))

def logisticRegression(X, weight, bias):
    #Return logistic regression evaluation
    return sigmoid(X.dot(weight)+bias)

def classificationAccuracy(y, y_p):
    #Accuracy of the classification
    return np.mean(y == y_p)

# Inside donut construction
theta = 2*np.pi*np.random.random(N)
R1 = R_inside + np.random.randn(N)
X_inside = np.c_[R1 * np.cos(theta), R1 * np.sin(theta)]
# Outside donut construction
theta = 2*np.pi*np.random.random(N)
R2 = R_outside + np.random.randn(N)
X_outside = np.c_[R2 * np.cos(theta), R2 * np.sin(theta)]

# X corrdinates
X = np.r_[X_inside,X_outside]
# y labels
y = np.r_[np.zeros(N),np.ones(N)]

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# One
unitvector = np.ones(N*2)
radius = np.sqrt(X[:,0]**2+X[:,1]**2)
X2 = np.c_[unitvector,radius,X]
D = 2


