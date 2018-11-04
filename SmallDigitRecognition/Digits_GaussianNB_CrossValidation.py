# -*- coding: utf-8 -*-
'''
Small Digit recognition
Author: Jungho Park
Date: March 1, 2017
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd

# Module to separate train and test data set
from sklearn.model_selection import train_test_split, validation_curve
# Import accuracy_score and confusion_matrix functions
from sklearn.metrics import accuracy_score, confusion_matrix
# Import cross validation tool
from sklearn.cross_validation import cross_val_score

# Dimensionality reduction using principal component analysis
from sklearn.naive_bayes import GaussianNB

# Read the small handwritten digit dataset
digitsData = pd.read_csv('digits_small.csv')
y_digits = digitsData['0']
X_digits = digitsData.drop('0',axis=1)

# Split the data to train and test dataset with 20% 
Xtrain, Xtest, ytrain, ytest = train_test_split(X_digits, y_digits, 
      random_state=0, test_size=0.2)

# Choose Gaussian Naive Bayes model
model = GaussianNB()

# Fit the model with the iris dataset
model.fit(Xtrain, ytrain)

# Evaluate the outcome for Xtest
y_fitted = model.predict(Xtest)

# Print the accuracy. It is 0.825. 
print("The accuracy of GaussianNB is %f" % (accuracy_score(ytest, y_fitted)))
confusionMat = confusion_matrix(ytest, y_fitted)
sns.heatmap(confusionMat, cbar=False, square=True, annot=True);
plt.xlabel('predicted digits'); plt.ylabel('true digits')

# Evaluate five-fold cross validation scores
cv_score = cross_val_score(model, X_digits, y_digits, cv=5)
print("Cross Validation Scores:",cv_score)
