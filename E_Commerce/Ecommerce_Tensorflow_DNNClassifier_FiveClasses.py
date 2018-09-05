'''
Neural netowrk constructed with Tensor flow applied to ecommerce_data with five class softmax terminal layer
https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv
Author: Jungho Park
Date: Oct 15, 2017
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os

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

def input(dataset):
  return dataset

if __name__ == '__main__':
  # Set the GPU off
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  # Get the data for the first two categories from ecommerce_data
  Xtrain,ytrain,Xtest,ytest = getWebVisitData('ecommerce_data.csv')
  # number of samples(N1) and features(D) of the input data
  N1,D = Xtrain.shape
  N2 = Xtest.shape[0]
  # Number of neurons for the first hidden layer
  M = D + 1
  # Number of classes for the softmax layer
  K = max(ytrain)+1
  
  feature_columns = [tf.feature_column.numeric_column("x", shape=[5])]
  # Build 3 layer DNN with 10, 20, 10 units respectively. How can I change this values? 
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                  hidden_units=[M, M+1], n_classes=K)
  # Define the TRAINING inputs, includes both the feature (DNN input end) and target (DNN output end)
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": input(Xtrain)}, #training_set.data
        y=input(ytrain), #training_set.target
        num_epochs=None,
        shuffle=True)
    
  #Fit model.
  print("Training classfier...")
  classifier.train(
        input_fn = train_input_fn,
        steps = 2000)

  #Define the TEST inputs, both feature and target
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":input(Xtest)},
        y=input(ytest),
        num_epochs=1,
        shuffle=False)

  #Evaluate accuracy after training
  accuracy_score = classifier.evaluate(
        input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
