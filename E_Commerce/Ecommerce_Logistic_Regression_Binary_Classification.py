'''
Binary logistic regression applied to ecommerce_data
https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv
Author: Jungho Park
Date: Oct 15, 2017
'''
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
# arr = np.arange(9).reshape((3,3))
# np.random.shuffle(arr)
# arr2 = shuffle(arr)

# repeat samples 9 times
# np.repeat(X1, 9, axis=0)

def sigmoid(x):
    # Sigmoid function
    # Sigmoid is 0.5 at x=0.5
    # Return: (0,1)
    return 1 / (1 + np.exp(-x))

def relu(x):
    # rectified linear unit
    return x*(x>0)

def tanh(x):
    # rectified linear unit
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def softmax(x):
    # softmax multiclass classification. Multinomial logistic regression
    tExp = np.exp(x)
    return tExp/tExp.sum(axis=1, keepdims=True)    

def forwardCalc(X, W, b):
    # Forward calculation for a simple logistic regression
    # X is the input matrix(nSamples,nFeatures)
    # The first feature of X is 1 and the first vector of W is the bias term
    # W is the weight matrix(nFeatures) 
    # b is the bias value
    # Returns vector(nSamples)
    return sigmoid(X.dot(W)+b)

def logisticRegression(X, weight, bias):
    #Return logistic regression evaluation
    return sigmoid(X.dot(weight)+bias)

def classificationAccuracy(y, y_p):
    #Accuracy of the classification
    return np.mean(y == y_p)

def costCrossEntropy(t,y):
    #Cross entropy cost function
    return -np.mean(t*np.log(y)+(1-t)*np.log(1-y))

def oneHotEncoding(y, K):
    # One hot encoding implementation
    # Returns NxK matrix of one-hot encoding from y label
    N = len(y)
    enc = np.zeros((N,K),dtype=np.int32)
    for i in range(N):
        enc[i,y[i]] = 1
    return enc

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
    # Evaluate the number of categories (K)
    K = len(set(y))
    # Construct one-hot encoding from the y label
    enc = oneHotEncoding(X[:,-1].astype(np.int32), K)
    X = np.c_[X[:,:-1],enc]
    # Assign the first 400 to Train data set
    Xtrain = X[:-100]
    ytrain = y[:-100]
    # Assign the remaining 100 to Test data set
    Xtest = X[-100:]
    ytest = y[-100:]
    # Return the train and test data set
    return Xtrain, ytrain, Xtest, ytest

def getFirst2Categories(fname):
    # Select the first two categories for a binary classifier
    Xtrain, ytrain, Xtest, ytest = getWebVisitData(fname)
    Xtrain = Xtrain[ytrain<=1]
    ytrain = ytrain[ytrain<=1]
    Xtest = Xtest[ytest<=1]
    ytest = ytest[ytest<=1]
    return Xtrain, ytrain, Xtest, ytest

def gradientDescent(Xtrain,ytrain,Xtest,ytest, W, bias, lRate, L2, max_iter=10000, print_interval=1000, tol = 1e-8):
    # Perform gradient descent
    lossTrain = []
    lossTest = []
    N = Xtrain.shape[0]
    ytrain_pred = sigmoid(Xtrain.dot(W))
    ytest_pred = sigmoid(Xtest.dot(W))
    for i in range(max_iter):
        # Print at 1000 iterations
        if i%print_interval == 0:
            costTrain = costCrossEntropy(ytrain,ytrain_pred)
            costTest = costCrossEntropy(ytest,ytest_pred)
            lossTrain.append(costTrain)
            lossTest.append(costTest)
            print("Cross Entry cost Train=%.4f, Test=%.4f" % (costTrain,costTest))
        # Calculate the difference between ytrain and ytrain predicted vectors            
        ydiff = (ytrain-ytrain_pred)
        # calculate correction term for W
        correctionW = lRate/N*(ydiff.dot(Xtrain) - L2*W)
        # calculate correction term for bias
        correctionBias = lRate*ydiff.sum() 
        if np.abs(correctionW).sum() < tol/lRate:
            break
        # Correct the W and bias
        W += correctionW
        bias += correctionBias
        # Recalculate the ytrain and ytest
        ytrain_pred = sigmoid(Xtrain.dot(W)+bias)
        ytest_pred = sigmoid(Xtest.dot(W)+bias)
    return np.round(ytrain_pred).astype(np.int32),np.round(ytest_pred).astype(np.int32), lossTrain, lossTest

if __name__ == '__main__':
    # Get the data for the first two categories from ecommerce_data
    Xtrain,ytrain,Xtest,ytest = getFirst2Categories('ecommerce_data.csv')
    # Obtain the shape of the np.array
    N1,D = Xtrain.shape
    N2,D = Xtest.shape
    # Set up a random weight matrix generated with normally distribution numbers
    # the bias term is ther first column
    W = np.random.randn(D)
    bias = 0
    # Learning rate
    learningRate = 0.001
    # L1 Regulization
    L1 = 0.0
    # Perform gradient descent operation
    ytrain_pred, ytest_pred, lossTrain, lossTest = gradientDescent(Xtrain,ytrain,Xtest,ytest,W,bias,learningRate, L1)
    print("Classification accuracy of the training set = %.4f" % (classificationAccuracy(ytrain, ytrain_pred)))
    print("Classification accuracy of the test set = %.4f" % (classificationAccuracy(ytest, ytest_pred)))
    
