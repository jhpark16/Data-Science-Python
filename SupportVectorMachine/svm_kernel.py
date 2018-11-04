'''
Soft boundary Support Vector Machine
Based on Cortes, C and Vapnik, V (1995) Support Vector Network

Author: Jungho Park
Date: Oct 20, 2017

Data: fer2013.csv - The facial expression recognition data from Kaggle challenge
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
'''

import time
import numpy as np
import pandas as pd
from numpy import linalg
from cvxopt import matrix, solvers
# Module to separate train and test data set
from sklearn.model_selection import train_test_split
# Import accuracy_score and confusion_matrix functions
from sklearn.metrics import accuracy_score, confusion_matrix

def getFacialDataTwoClass(fname):
  # Obtain the first two classes y==0 or y==1
  X = []
  y = []
  bHeader = True
  with open(fname,'r') as f:
    for line in f:
      # ignore the header
      if bHeader:
        bHeader = False
      else:
        # Split the line of the comma seperated value file with ','
        row = line.split(',')
        # The first column is the facial expression
        faceExp = int(row[0])
        # Only for facial expression 0 or 1 categories
        if faceExp==0 or faceExp==1:
          # Append the values to X and y
          X.append([int(it) for it in row[1].split()])
          y.append(faceExp)
    # Scale the X to [0,1) to prevent numerical calculation problems
    X = np.array(X)/255
    y = np.array(y)
  return X,y

def linear(X1, X2):
  # Linear kernel X1.dot(X2)
  return np.dot(X1, X2)

def polynomial(X1, X2, p=2):
  # Polynomial kernel
  return (1 + np.dot(X1, X2))**p

def rbf(X1, X2, sigma=4.0):
  # Gaussian kernel
  m1 = 1 if X1.ndim == 1 else X1.shape[0]
  m1f = X1.shape[-1]
  m2 = 1 if X2.ndim == 1 else X2.shape[0]
  m2f = X2.shape[-1]
  rbf = np.empty((m1,m2),dtype=float)
  x_1 = X1.reshape(m1,m1f)
  x_2 = X2.reshape(m2,m1f)
  for j in range(m1):
    for i in range(m2):
      tVec = x_1[j]-x_2[i]
      rbf[j][i] = np.exp(-tVec.dot(tVec) / (2 * (sigma ** 2)))
  return rbf

class SVM(object):
  # Support Vector Machine implementation using cvxopt qp optimizer
  def __init__(self, kernel=linear, C=None, p=2, sigma=4.0):
    # Initialize hyperparameters
    self.kernel = kernel
    self.sigma = sigma
    self.p = p
    self.C = C
    if self.C is not None: self.C = float(self.C)

  def fit(self, X, y):
    # Fit the model to the train dataset
    nSamples, nFeatures = X.shape
    # kernel setting
    if self.kernel==linear:
      K = linear(X, X.T)
    elif self.kernel==polynomial:
      K = polynomial(X, X.T, self.p)
    else:
      K = rbf(X, X, self.sigma)
    self.K = K
    # CVXOPT: Evaluate a Lambda vector satisfying all the constraints.
    # CVXOPT: solvers.qp minimizes (1/2)Lamnda^T P Lamnda + q^T Lambda
    # Cortes and Vapnik (1995) Equation(66 and following equations) 
    # requires the maximization of -(1/2)Lamnda^T D Lamnda + Lambda.
    # It is the minimization of (1/2)Lamnda^T D Lamnda - Lambda.
    # D (which is P in CVXOPT) is np.outer(y,y)*kernel
    # q is -1
    P = matrix(np.outer(y,y) * K)
    q = matrix(np.ones(nSamples) * -1)
    # CVXOPT: A * X = b
    # Cortes and Vapnik(1995): Lambda^T * Y = 0 (Equation 61)
    A = matrix(y.reshape(1,-1), tc='d')
    b = matrix(0.0)

    # CVXOPT constraint: G * Lambda <= h
    # The constraints of Cortes and Vapnik 0 <= Lambda <= C are
    # -Lambda <=0 (G_upper) and Lambda <=C (G_lower)
    if self.C is None:
        G = matrix(np.diag(np.ones(nSamples) * -1))
        h = matrix(np.zeros(nSamples))
    else:
        G_upper = np.diag(np.ones(nSamples) * -1)
        G_lower = np.identity(nSamples)
        G = matrix(np.vstack((G_upper, G_lower)))
        h_upper = np.zeros(nSamples)
        h_lower = np.ones(nSamples) * self.C
        h = matrix(np.hstack((h_upper, h_lower)))

    # solve QP problem
    self.solution = solvers.qp(P, q, G, h, A, b)

    # Obtain the alpha (Equation 12)
    alpha = np.ravel(self.solution['x'])
    self.alpha_ = alpha
    
    # A large alpha values are support vectors
    s_v = alpha > 1e-5
    self.sv_ = s_v
    ind = np.arange(len(alpha))[s_v]
    self.ind = ind
    self.alpha = alpha[s_v]
    self.sv_X = X[s_v]
    self.sv_y = y[s_v]
    print("%d support vectors from %d samples" % (len(self.alpha),nSamples))

    # Calculate W using equation 66
    self.W = (self.sv_y*self.alpha).dot(self.sv_X)       
    # Calculate b
    self.sv_K = K[:,s_v][ind,:]
    self.b = (np.sum(self.sv_y)-np.sum(self.alpha*self.sv_y*self.sv_K)
      )/len(self.sv_y)
    return
    
  def predict(self, X):
    # Predict the outcome with the given X
    if self.kernel == linear:
      self.y_predict = X.dot(self.W) + self.b
    elif self.kernel == polynomial:
      self.y_predict = np.sum(self.alpha*self.sv_y*
        self.kernel(X, self.sv_X, self.p), axis=-1) + self.b
    else:
      K1 = self.kernel(X, self.sv_X, self.sigma)
      self.y_predict = np.sum(self.alpha*self.sv_y*K1, axis=-1) + self.b
    return np.sign(self.y_predict)

if __name__ == "__main__":
  # Read the first two classes of the facial expression data
  # The data is already normalized to [0,1]. There is no need for scaling.
  X,y = getFacialDataTwoClass('fer2013.csv')
  # Seperate two classes
  X_Class0 = X[y==0]
  X_Class1 = X[y==1]
  # Adjust for unblanced samples
  # Increase the number of smaller class by repeating the the samples 
  if X_Class0.shape[0]>X_Class1.shape[0]:
    nRepeat = int(round(X_Class0.shape[0]/X_Class1.shape[0]))
    X_Class1 = np.repeat(X_Class1, nRepeat, axis=0)
  else:
    nRepeat = int(round(X_Class1.shape[0]/X_Class0.shape[0]))
    X_Class0 = np.repeat(X_Class0, nRepeat, axis=0)
  # Update X and y labels
  X = np.r_[X_Class0, X_Class1]
  y = np.array([-1]*len(X_Class0) + [1]*len(X_Class1))
  
  # Split the data to train and test dataset with 20% 
  Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
        random_state=0, test_size=0.2)
  
  # Choose Gaussian Naive Bayes model
  model = SVM(kernel=linear, C=8.0)
  #model = SVM(kernel=rbf, C=1.0, sigma=1)
  
  # Fit the model with the iris dataset
  t1 = time.time()
  model.fit(Xtrain, ytrain)
  t2 = time.time()
  print("Training takes %f secs" % (t2-t1))
  # Evaluate the outcome for Xtest
  y_fitted = model.predict(Xtest)
  t1 = time.time()
  print("Prediction takes %f secs" % (t1-t2))
  
  # The accuracy of SVC(kernel='linear') is 0.899798 
  # The accuracy of SVC(kernel='rbf') is 0.698887
  # The accuracy of SVC(kernel='rbf', 
  # gamma=0.001, 0.005, 0.01, 0.1 are 0.760628, 0.933198, 0.980769, 1.0 
  # The accuracies of SVM(kernel=linear, C=0.01, 0.1, 1.0, 5.0, 10.0, 100.0) 
  # are 0.7707, 0.846660, 0.904352, 0.913968, 0.915992, 0.915992
  # The accuracies of SVM(kernel=rbf, C=1.0
  # sigma=100, 10, 1
  # are 0.613866, 0.934211, 1.0000
  print("The accuracy is %f" % (accuracy_score(ytest, y_fitted)))
  
