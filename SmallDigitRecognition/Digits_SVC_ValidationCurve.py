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

# Support Vector Machine for classification
from sklearn.svm import SVC

# Read the small handwritten digit dataset to a dataframe using panda
digitsData = pd.read_csv('digits_small.csv')
y_digits = digitsData['0']
X_digits = digitsData.drop('0',axis=1)

# The range of 20 gamma hyper-paramters to be evaluated
gamma_range = 10**(np.linspace(-6, -1, 20))

# Calculate the validation curve
train_scores, test_scores = validation_curve(SVC(kernel='rbf'), 
    X_digits, y_digits, cv=10, scoring="accuracy", 
    param_name="gamma", param_range=gamma_range)
# Calculate the mean scores of cross validation train and test dataset for each gamma value
mean_train_scores = np.mean(train_scores, axis=1)
mean_test_scores = np.mean(test_scores, axis=1)

#Plot the accuracy scores of train and test dataset vs gamma value
# The best cross validation score is about 0.98 at gamma = 0.0014
plt.xlabel(u"γ"); plt.ylabel("Accuracy Score");
plt.title("SVM Accuracy vs gamma value");
plt.semilogx(gamma_range, mean_train_scores, label="Train");
plt.semilogx(gamma_range, mean_test_scores, label="Test");
plt.legend();
plt.show()
