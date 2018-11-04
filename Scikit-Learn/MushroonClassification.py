'''
Mushroom classification (poisonous or not) using scikit-learn
Author: Jungho Park
Date: February 12, 2017
'''
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost

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

# Read mushroom data from Kaggle challenge
mushroom = pd.read_csv("mushrooms.csv")

# Convert string(label) features to numeric features
labelEncoder = preprocessing.LabelEncoder()
savedEncoders = []
for column in mushroom.columns:
  mushroom[column] = labelEncoder.fit_transform(mushroom[column])
  savedEncoders.append(labelEncoder)

Xmushroom = mushroom[mushroom.columns ^ pd.Index(['class'])]
ymushroom = labelEncoder.fit_transform(mushroom['class'])

# Split the data into train and test sets. Test set is the 15% of the data.
dataTrain, dataTest = train_test_split(mushroom, test_size = 0.15) 
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

# Print out the importance of the features
xgboost.plot_importance(models[2][0])

# Dimensionality reduction
model = PCA(n_components=7)

# Fit the model with the Xmushroom data
model.fit(Xmushroom)

# Convert the 4-dimensional iris dataset to 2-D dataset
X_dat_2D = model.transform(Xmushroom)

# Add the Reduced 2D dataset to the dataframe for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_dat_2D[:, 0], X_dat_2D[:, 1], X_dat_2D[:, 2], cmap=ymushroom)
plt.show()

mushroom['X_Red1'] = X_dat_2D[:, 0]
mushroom['X_Red2'] = X_dat_2D[:, 1]
mushroom['X_Red3'] = X_dat_2D[:, 2]
# Plot the reduced dataset of the dataframe using seaborn package 
sns.lmplot('X_Red1','X_Red2',data=mushroom, fit_reg=False, hue='class')
sns.lmplot('X_Red1','X_Red3',data=mushroom, fit_reg=False, hue='class')

# Plot the PCA result, the first three components are significant
plt.plot(np.linspace(1,7,7),model.explained_variance_ratio_)
plt.show()

