'''
Neural netowrk constructed with Tensor flow applied to ecommerce_data with five class softmax terminal layer
https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv
Author: Jungho Park
Date: Oct 15, 2017
'''
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

def getWebVisitData(fname):
  # Read web visit data using pandas's read_csv with header
  # Returns Xtrain, ytrain, Xtest and ytest
  dFrame = pd.read_csv(fname)
  # Covnert it to matrix format from data frame
  mat = dFrame.as_matrix()
  # randomize the data order
  np.random.shuffle(mat)
  # Assign all the features to X
  X = mat[:,:-1]
  # The last column is the label
  y = mat[:,-1].astype(np.int32)
  # Assign the first 316 to Train data set
  Xtrain = X[:-100]
  ytrain = y[:-100]
  # Assign the remaining 100 to Test data set
  Xtest = X[-100:]
  ytest = y[-100:]
  # Return the train and test data set
  return Xtrain, ytrain, Xtest, ytest

if __name__ == '__main__':
  # Get the data for the first two categories from ecommerce_data
  Xtrain,ytrain,Xtest,ytest = getWebVisitData('ecommerce_data.csv')
  # number of samples(N1) and features(D) of the input data
  N1,D = Xtrain.shape
  N2 = Xtest.shape[0]
  # Number of neurons for the first hidden layer
  M = D + 1
  # Number of classes for the softmax layer
  K = max(ytrain)+1
  
  model = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=2000)
  #activation='logistic', solver='lbfgs', tol=1e-10, batch_size=16, 
  # alpha=0.000, learning_rate='constant', learning_rate_init=0.01, 
  model.fit(Xtrain, ytrain)
  # The accuracy reaches 0.975 after 3000 epochs
  train_result = model.score(Xtrain,ytrain)
  test_result = model.score(Xtest,ytest)
  # 0.12571, 0.95 0.9675(K=8, epch=2000)
  # 0.15584, 0.92 0.9725(K=7, epch=2000)
  # 0.19238, 0.92 0.975(K=6, epch=2000)
  # 0.28027, 0.93 0.9825 (K=6,6, epch=4000)
  # 0.18958, 0.96 0.9825 (K=6,12, epch=4000)
  print("Accuray: Train=%.5f, Test=%.5f" %(train_result, test_result))

