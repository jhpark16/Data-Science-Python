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
  
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(M, activation=tf.nn.sigmoid),
      tf.keras.layers.Dense(M+1, activation=tf.nn.sigmoid),
  #    tf.keras.layers.Dense(M, activation=tf.nn.relu),
  #    tf.keras.layers.Dense(M, activation=tf.nn.tanh),
      tf.keras.layers.Dense(K, activation=tf.nn.softmax)
  ])
  #model.compile(optimizer='adam', loss='categorical_crossentropy',
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  start = time.time()
  model.fit(Xtrain, ytrain, verbose=1, batch_size=8, epochs=100) # 10000 is more than enough
  end = time.time()
  print("Running Time: %.2f secs" %(end - start))
  # The accuracy reaches 0.975 after 3000 epochs
  test_result = model.evaluate(Xtest,ytest)
  # 0.12571, 0.95 0.9675(K=8, epch=2000)
  # 0.15584, 0.92 0.9725(K=7, epch=2000)
  # 0.19238, 0.92 0.975(K=6, epch=2000)
  # 0.28027, 0.93 0.9825 (K=6,6, epch=4000)
  # 0.60164, 0.91 0.9975 (K=6,7, epch=10000)
  # 0.18958, 0.96 0.9825 (K=6,12, epch=4000)
  print("Test loss=%.5f, accuracy=%.5f" %(test_result[0],test_result[1]))
  
