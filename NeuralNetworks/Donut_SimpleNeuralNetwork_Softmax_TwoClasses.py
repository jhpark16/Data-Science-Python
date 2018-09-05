# -*- coding: utf-8 -*-
"""
Evaluate the performance of a simple neural network (with a 3 neuron hidden layer
 and softmax final layer) applied to a donut dataset. Unlike logsitic regression,
this simple neural network does not need a calculated radius feature for the donut dataset.
The training takes a lot longer than logistic regression.

Author: Jungho Park
Date: October 31 2016
"""
import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork(object):
  # Logistic regression for binary classification
  def __init__(self, nFeatures, nHidden, nCategories):
    # Initialize the model with
    self.D = nFeatures # number of features
    self.M = nHidden # number of hidden layers
    self.K = nCategories # number of categories for softmax
    # Fill the weights and bias with random numbers
    self.W1 = np.random.randn(self.D, self.M)*0.5
    self.b1 = np.random.randn(self.M)*0.5
    self.W2 = np.random.randn(self.M, self.K)*0.5
    self.b2 = np.random.randn(self.K)*0.5
    return
  
  def grads(self, X, y_target_onehot, y_hat, L2):
    # Gradient calculations for the hidden and final layers
    m = X.shape[0]
    # Calculate the delta Z for the final layer
    self.dZ2 = y_hat - y_target_onehot
    # Calculate dJ/dW
    self.dW2 = self.A1.T.dot(self.dZ2)/m + L2*self.W2/m
    # Calculate dJ/db
    self.db2 = self.dZ2.T.sum(axis=1)/m
    # For sigmoid and softmax activations (g(x)' = A1*(1-A1))
    # For tanh activation (g(x)' = (1-A1^2))
    self.dZ1 = self.dZ2.dot(self.W2.T)* self.A1 * (1-self.A1)
    # Calculate dJ/dW
    self.dW1 = X.T.dot(self.dZ1)/m + L2*self.W1/m
    # Calculate dJ/db
    self.db1 = self.dZ1.T.sum(axis=1)/m
    return self.dW1, self.db1, self.dW2, self.db2
  
  def forwardCalc(self, X):
    # Forward calculation for a simple neural network
    # X is the input matrix(nSamples, nFeatures)
    # W1 is the weight matrix(nFeatures, nNeurons) 
    # b1 is the bias value(nNeurons)
    # W2 is the weight matrix(nNuerons, nSoftmaxCategories) 
    # b2 is the bias value(nSoftmaxCategories)
    # Returns vector(nSoftmaxCategories)
    # The hidden layer uses sigmoid activation
    self.Z1 = X.dot(self.W1) + self.b1
    self.A1 = sigmoid(self.Z1)
    # The final layer uses softmax
    self.Z2 = self.A1.dot(self.W2) + self.b2
    self.A2 = softmax(self.Z2)
    return self.A2
  
  def fit(self, X, y, epochs=100000, alpha=0.2, L2=0.0):
    # Fit the model to the data X, y
    # alpha is the learning rate
    # L2 is the L2 regulization term
    K = int(np.max(y)+1)
    y_target_onehot = self.oneHotEncoding(y, K)
    costs = []
    y_hat = self.forwardCalc(X)
    for i in range(epochs):
      dW1, db1, dW2, db2 = self.grads(X, y_target_onehot, y_hat, L2)
      # Update W1, b1, W2, b2 using the calculated gradient
      self.W2 -= alpha *dW2
      self.b2 -= alpha *db2
      self.W1 -= alpha *dW1
      self.b1 -= alpha *db1
      y_hat = self.forwardCalc(X)
      if i % 100 == 0:
        # Calculate the cross entropy cost
        costVal = costCrossEntropyVector(y_target_onehot, y_hat)
        # Calculate the accuracy
        accuracy = self.accuracy(X,y)
        print("%6d iterations: Cost = %.8f, Accuracy:%.6f" % (i, costVal, accuracy))
        costs.append(costVal)
    return costs
  
  def oneHotEncoding(self, y, K):
    # One hot encoding implementation
    # Returns NxK matrix of one-hot encoding from y label
    N = len(y)
    enc = np.zeros((N,K),dtype=np.int32)
    for i in range(N):
        enc[i,int(y[i])] = 1
    return enc
  
  def predict(self, X):
    # Calculate the prediction of the current model
    y_hat = self.forwardCalc(X)
    return np.argmax(y_hat, axis=1)
  
  def accuracy(self, X, y):
    # Calculate the accuracy of the current model
    prediction = self.predict(X)
    return np.mean(y==prediction)



def sigmoid(x):
  # Sigmoid function
  # Sigmoid is 0.5 at x=0.5
  # Return: (0,1)
  return 1 / (1 + np.exp(-x))

def softmax(x):
    # softmax multiclass classification. Multinomial logistic regression
    tExp = np.exp(x)
    return tExp/tExp.sum(axis=1, keepdims=True)

def costCrossEntropyVector(y_target, y_hat):
  #Cross entropy cost function
  return -(y_target*np.log(y_hat)).mean()



if __name__ == '__main__':
  # Number of data points for inside and outside donuts
  N = 400
  
  # mean radius of inside and outside donuts
  R_inside = 3
  R_outside = 8
  
  # Create a donut dataset
  # Inside donut construction
  theta = 2*np.pi*np.random.random(N)
  R1 = R_inside + np.random.randn(N)
  X_inside = np.c_[R1 * np.cos(theta), R1 * np.sin(theta)]
  # Outside donut construction
  theta = 2*np.pi*np.random.random(N)
  R2 = R_outside + np.random.randn(N)
  X_outside = np.c_[R2 * np.cos(theta), R2 * np.sin(theta)]
  
  # X corrdinates of the donut
  X = np.r_[X_inside,X_outside]
  # y labels of the donut
  y = np.r_[np.zeros(N),np.ones(N)]
  
  plt.scatter(X[:,0], X[:,1], c=y)
  plt.show()
  
  # number of features of the input data
  D = 2
  # Number of neurons for the first hidden layer
  M = 3
  # Number of classes for the softmax layer
  K = 2
  model = SimpleNeuralNetwork(D, M, K)
  # Simple two layer neural network setup
  
  # Train the model with the train set
  costs = model.fit(X,y,epochs=100000)
  # Calculate model accurcy with X
  accuracy = model.accuracy(X,y)
  # print accuracy
  print("Accuracy = %.5f" % (accuracy))
