# -*- coding: utf-8 -*-
'''
Learning Curve (Training and cross-validation scores vs # of training samples)
Classification of the Titanic survival using XGBoost classifier
Author: Jungho Park
Date: December 1, 2016
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBClassifier

# Read training and test data of Titanic survivors
Train = pd.read_csv('titanic/train.csv')
Test = pd.read_csv('titanic/test.csv')
# Fill the missing ages with normally distributed random ages 
# based on the mean and std
Train["Age"][np.isnan(Train["Age"])] = (
    Train["Age"].mean()+np.random.randn(Train["Age"].isnull().sum())
    *Train["Age"].std())
Train["Age"] = Train["Age"].astype(int)
Test["Age"][np.isnan(Test["Age"])] = (
    Test["Age"].mean()+np.random.randn(Test["Age"].isnull().sum())
    *Test["Age"].std())
Test["Age"] = Test["Age"].astype(int)

# Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# Fill the two missing data with Southampton
Train["Embarked"].fillna('S', inplace=True)
# Change the category type of Embarked to int
Train['Embarked'] = Train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
Test['Embarked'] = Test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Fill the missing Fare with a mean value
Test["Fare"].fillna(Test["Fare"].median(), inplace=True)

# When the cabin number is avaialbe, Cabin feature is one.
# Otherwise it is zero.
Train["Cabin"] = Train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
Test["Cabin"] = Test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Map sex to number
Train['Sex'] = Train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
Test['Sex'] = Test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Group the age into five groups
Train["Age"] = Train["Age"].apply(lambda x: 0 if x<=14 else 1 if x<=32 else 2 
     if x<=48 else 3 if x<=64 else 4)
Test["Age"] = Test["Age"].apply(lambda x: 0 if x<=14 else 1 if x<=32 else 2 
     if x<=48 else 3 if x<=64 else 4)
# Group the fare into four groups
Train["Fare"] = Train["Fare"].apply(lambda x: 0 if x<=7.91 else 1 if x<=14.454 else 2 
     if x<=31 else 3)
Train['Fare'] = Train['Fare'].astype(int)
Test["Fare"] = Test["Fare"].apply(lambda x: 0 if x<=7.91 else 1 if x<=14.454 else 2 
     if x<=31 else 3)
Test['Fare'] = Test['Fare'].astype(int)

# Delete Name feature. It is not useful.
del Train['Name']
del Test['Name']
# Delete ticket feature. It is not useful.
del Train['Ticket']
del Test['Ticket']
# Delete Siblings and Souses. It is not used.
del Train['SibSp']
del Test['SibSp']
# Delete Parents/Children. It is not used.
del Train['Parch']
del Test['Parch']
# Delete PassengerId. It is not useful.
del Train['PassengerId']
del Test['PassengerId']

# Setup Xtrain, ytrain and XTest
ytrain = Train['Survived']
del Train['Survived']
Xtrain = Train
Xtest = Test

# Setup cross validation method
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# The range of train sizes to be tested
train_sizes=np.linspace(.1, 1.0, 20)

# XGBoost classifier
model = XGBClassifier()

# Evaluate the learning curve (scores) as a function of trainng sample size
train_sizes, train_scores, test_scores = learning_curve(model, Xtrain, ytrain, 
          cv=cv, n_jobs=4, train_sizes=train_sizes)
# Calculate the average
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean  = np.mean(test_scores, axis=1)

# Plot the learning curve
plt.figure()
# Plot title
plt.title("Learning Curve with XGB Classifier (Titanic Survival)")
# X and Y axis labels
plt.xlabel("# of training examples")
plt.ylabel("Score")
# Add grid
plt.grid()

# Plot the training score vs # of training samples
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
# Plot the cross validation score vs # of training samples
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")
# Add the legend at the best location
plt.legend(loc="best")
plt.show()

