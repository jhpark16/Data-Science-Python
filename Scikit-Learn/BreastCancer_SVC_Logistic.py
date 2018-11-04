'''
Breast Cancer classification using scikit-learn SVC
Author: Jungho Park
Date: February 12, 2017
'''
import numpy as np
import pandas as pd

# Module to separate train and test data set
from sklearn.model_selection import train_test_split
# For LabelEncoder
from sklearn import preprocessing
# Support Vector Machine model
from sklearn.svm import SVC
# Logistic Regression model
from sklearn.linear_model import LogisticRegression
# XGBooster classifier
from xgboost import XGBClassifier
# For accuracy score evaluation
from sklearn.metrics import accuracy_score
# Dimensionality reduction using principal component analysis
from sklearn.decomposition import PCA  

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read Breast cancer data 
dataFrame = pd.read_csv("breast-cancer-wisconsin.csv")
# Drop sample_id because it is not relevant
dataFrame.drop('sample_id', axis=1, inplace=True)
# Replace missing data with -99
dataFrame.replace('?', -99, inplace=True)
# Convert the data to float type
dataFrame = dataFrame.astype(np.float)
# Split the data into train and test sets. Test set is the 20% of the data.
dataTrain, dataTest = train_test_split(dataFrame, test_size = 0.2) 
# ytrain is the class column
ytrain = dataTrain['class']
# Xtrain is the feature columns (without class column)
Xtrain = dataTrain[dataTrain.columns ^ pd.Index(['class'])]

# ytest is the class column
ytest = dataTest['class']
# Xtest is the feature columns (without class column)
Xtest = dataTest[dataTest.columns ^ pd.Index(['class'])]

# Setup models and the name of the model
models = [(SVC(kernel='rbf', random_state=0),'SVC rbf'),
          (SVC(kernel='linear', random_state=0),'SVC linear'),
          (XGBClassifier(),'XGboost classifier'), 
          (LogisticRegression(),'Logistic Regression')]
# Evaluate all the models
for model in models:
  # Fit the model to the data
  model[0].fit(Xtrain, ytrain)
  # Evaluate the accuracy
  score = accuracy_score(ytest, model[0].predict(Xtest))
  # SVC rbf and xgboost have a very high score
  print('The accurancy of %s is %f' % (model[1], score))
  
""" The result is 
The accurancy of SVC rbf is 0.971429
The accurancy of SVC linear is 0.964286
The accurancy of XGboost classifier is 0.950000
The accurancy of Logistic Regression is 0.971429
"""
# Print out the importance of the features
xgboost.plot_importance(models[2][0])

# Dimensionality reduction
model = PCA(n_components=2)

# Fit the model with the Xmushroom data
Xdata = dataFrame[dataFrame.columns ^ pd.Index(['class'])]
ydata = dataFrame['class']
model.fit(Xdata)

# Convert the 4-dimensional iris dataset to 2-D dataset
X_dat_2D = model.transform(Xdata)

#
dataFrame['X_Red1'] = X_dat_2D[:, 0]
dataFrame['X_Red2'] = X_dat_2D[:, 1]

# Plot the reduced dataset of the dataframe using seaborn package 
sns.lmplot('X_Red1','X_Red2',data=dataFrame, fit_reg=False, hue='class')
