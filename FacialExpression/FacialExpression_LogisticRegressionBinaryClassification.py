'''
Data reader fro the facial recognition data from Kaggle challenge
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
'''
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class LogisticRegBinaryClassification(object):
  # Logistic regression for binary classification
  def __init__(self):
    # Init function -- do nothing
    pass

  def fit(self, X, y, epochs=200000, alpha=1.e-6, L2=0.0):
    # Fit the model to the data X, y
    # N is the number of samples, K is the number of feeatures
    N, K = X.shape
    self.W = np.random.randn(K) / np.sqrt(K)
    self.b = 0
    costs = []
    y_hat = forwardCalc(X, self.W, self.b)
    for i in range(epochs):
      self.W -= alpha*(y_hat-y).dot(X)+L2*self.W
      self.b -= alpha*(y_hat-y).sum()+L2*self.b
      y_hat = forwardCalc(X, self.W, self.b)
      if i % 20 == 0:
        costVal = costCrossEntropy(y,y_hat)
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



if __name__ == '__main__':
  # Application of binary logistic regression on facial expressions
  # Read the data
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
  y = np.array([0]*len(X_Class0) + [1]*len(X_Class1))
  #
  # Shuffle the data
  X,y = shuffle(X,y)
  # Divide the data to Xtrain, ytrain, Xtest, and ytest
  nTrain = int(y.shape[0]*0.8)
  Xtrain, ytrain = X[:nTrain],y[:nTrain]
  Xtest, ytest = X[nTrain:],y[nTrain:]
  # Setup the logistic regression binary classification model
  model = LogisticRegBinaryClassification()
  # Train the model with the train set.
  # No significant improvement can be achieved after 150000 iterations
  costs = model.fit(Xtrain,ytrain,epochs=100000)
  # Calculate model accurcy
  accuracyTrain = model.accuracy(Xtrain,ytrain)
  accuracyTest = model.accuracy(Xtest,ytest)
  # The accuracy after 100000 iterations
  # Train = 0.87215 and Test = 0.81832
  print("Accuracy: Train = %.5f, Test = %.5f" % (accuracyTrain, accuracyTest))

