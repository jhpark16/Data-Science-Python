# -*- coding: utf-8 -*-
"""
Evaluate the performance of binary logistic regression with an XOR type problem
Author: Jungho Park
Date: October 14 2016
"""
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(object):
  # Logistic regression for binary classification
  def __init__(self):
    # Init function -- do nothing
    pass

  def fit(self, X, y, epochs=100, alpha=1.e-4, L2=0.0):
    # Fit the model to the data X, y
    # N is the number of samples, K is the number of feeatures
    N, K = X.shape
    # Initialize W with normally distributed random values
    self.W = np.random.randn(K) / np.sqrt(K)
    self.b = 0
    costs = []
    y_hat = forwardCalc(X, self.W, self.b)
    for i in range(epochs):
      self.W -= alpha*(y_hat-y).dot(X)+L2*self.W
      self.b -= alpha*(y_hat-y).sum()+L2*self.b
      y_hat = forwardCalc(X, self.W, self.b)
      if i % 1 == 0:
        # Calculate the cross entropy cost
        costVal = costCrossEntropy(y,y_hat)
        # Calculate the accuracy
        accuracy = self.accuracy(X,y)
        print("%6d iterations: Cost = %.5f, Accuracy:%.5f" % (i, costVal, accuracy))
        costs.append(costVal)
    return costs
  
  def predict(self, X):
    # Calculate the prediction of the current model
    y_hat = forwardCalc(X, self.W, self.b)
    return np.round(y_hat)

  def accuracy(self, X, y):
    # Calculate the accuracy of the current model
    prediction = self.predict(X)
    return np.mean(y==prediction)



def forwardCalc(X, W, b):
  # Forward calculation for a simple logistic regression
  # X is the input matrix(nSamples,nFeatures)
  # The first feature of X is 1 and the first vector of W is the bias term
  # W is the weight matrix(nFeatures) 
  # b is the bias value
  # Returns vector(nSamples)
  return sigmoid(X.dot(W) + b)

def sigmoid(x):
  # Sigmoid function
  # Sigmoid is 0.5 at x=0.5
  # Return: (0,1)
  return 1 / (1 + np.exp(-x))
  
def costCrossEntropy(y, y_hat):
  #Cross entropy cost function
  return -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)).mean()



if __name__ == '__main__':
# Number of data points for each group
N = 100
SepDistance = 2.0
# Create an XOR type dataset
x1 = SepDistance + np.random.randn(N)
y1 = SepDistance + np.random.randn(N)
X1 = np.c_[x1,y1]
x1 = -SepDistance + np.random.randn(N)
y1 = -SepDistance + np.random.randn(N)
X2 = np.c_[x1,y1]
x1 = SepDistance + np.random.randn(N)
y1 = -SepDistance + np.random.randn(N)
X3 = np.c_[x1,y1]
x1 = -SepDistance + np.random.randn(N)
y1 = SepDistance + np.random.randn(N)
X4 = np.c_[x1,y1]

# Combine all X
X = np.r_[X1,X2,X3,X4]
# y labels of the XOR type data set
y = np.r_[np.zeros(N*2),np.ones(N*2)]

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# Add a unit vector
unitvector = np.ones(N*4)
# add a column of x*y
xy = X[:,0] * X[:,1]
# Combine all columns
X = np.c_[X,unitvector,xy]
# Logistic regression setup
model = LogisticRegression()
# Train the model with the train set
costs = model.fit(X,y,epochs=100)
# Calculate model accurcy with X
accuracy = model.accuracy(X,y)
# print accuracy
print("Accuracy = %.5f" % (accuracy))
# Predicted y
y_hat = model.predict(X)

# Plot the predicted classification
plt.scatter(X[:,0], X[:,1], c=y_hat)
plt.show()
