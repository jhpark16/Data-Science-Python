# -*- coding: utf-8 -*-
"""
Decision tree class implementation
Classification of the iris dataset using the decision tree classifier
Author: Jungho Park
Date: April, 2017
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionTree(object):
  # Decision Tree Classification
  def __init__(self, max_depth=5, min_samples_split=2):
    # The maximum depth of the decision tree
    self.max_depth = max_depth
    # The minimum number of samples required to split a node
    self.min_samples_split = min_samples_split 
    
  def gini(self, group_list, classes):
    # Evaluate gini index
    # Evaluate the total number of items in the group_list
    total_items = sum([len(group) for group in group_list])
    # Loop through all groups and calculate the gini index
    gini_value = 0.0
    for group in group_list:
      size = float(len(group))
      # If the size is greater than 0
      if size > 0:
        # Initialize count 
        count = dict(zip(classes,np.zeros((classes.size))))
        # Count the number of items
        for item in group:
          count[item[-1]] += 1
        score = 0.0
        # Add up the scores
        for item in classes:
          score += (count[item]/size)**2
        # Multiply the score with the fraction of each group
        gini_value += (1.0 - score) * (size / total_items)
    return gini_value
  
  # split the datalist into left and right branches based on
  # the index(column) and the value
  def binary_split(self, datalist, idx, value):
    l_branch, r_branch = list(), list()
    for list1 in datalist:
      if list1[idx] < value: l_branch.append(list1)
      else: r_branch.append(list1)
    return l_branch, r_branch
  
  def optimum_split(self, datalist1):
    # Evaluate the optimum split of the data
    # Initialize gscore with a large number
    opt_score = 1e30
    # Evaluate the classes of the data
    datalist = np.array(datalist1)
    classes = np.unique(datalist[:,-1]).astype(int)
    shape = datalist.shape
    # Loop through the list item except the class value (last item)
    for list1 in datalist:
      for idx in range(shape[1]-1):
        group_list = self.binary_split(datalist, idx, list1[idx])
        score = self.gini(group_list, classes)
        if score < opt_score:
          opt_score = score
          opt_iFeature, opt_value, opt_group_list = idx, list1[idx], group_list
    return {'iFeature':opt_iFeature, 'threshold':opt_value, 'node':opt_group_list}

  # Set the node value with the largest category in the group
  def group2value(self, group_list):
    # Collect the class list of the group
    class_values = [int(list1[-1]) for list1 in group_list]
    # return the dominant value among the group
    return max(set(class_values), key=class_values.count)

  # Split the binary tree or terminate the brach
  # based on the max_depth and min_samples_split values
  def split(self, btree, depth, max_depth, min_samples_split):
    l_branch, r_branch = btree['node']
    del(btree['node'])
    # If either left or right branch is empty, evaluate the representative value of 
    # the collection and set the node with the value
    if not l_branch or not r_branch:
      btree['l_branch'] = btree['r_branch'] = self.group2value(l_branch + r_branch)
      return
    # If the tree reached the maximum depth, convert the group list to a value
    # and return
    if depth >= max_depth:
      btree['l_branch'], btree['r_branch'] = self.group2value(l_branch), self.group2value(r_branch)
      return
    # If the left brach has more entries than min_samples_size, 
    # evaluate the optimum split of the branch and 
    # call split at the next depth level
    if len(l_branch) > min_samples_split:
      btree['l_branch'] = self.optimum_split(l_branch)
      self.split(btree['l_branch'], depth+1, max_depth, min_samples_split)
    else:
      # if not, convert the branch into a value leaf
      btree['l_branch'] = self.group2value(l_branch)
    # If the right brach has more entries than min_samples_size, 
    # evaluate the optimum split of the branch and 
    # call split at the next depth level
    if len(r_branch) > min_samples_split:
      btree['r_branch'] = self.optimum_split(r_branch)
      self.split(btree['r_branch'], depth+1, max_depth, min_samples_split)
    else:
      # if not, convert the branch into a value leaf
      btree['r_branch'] = self.group2value(r_branch)
    return
  
  # Fit the decision tree using training data
  def fit(self, Xtrain, ytrain):
    # Combine X matrix and y vector
    train = np.c_[Xtrain,ytrain]
    # Evaluate the first binary split
    root = self.optimum_split(train)
    # Split the remaining branches 
    self.split(root, 1, self.max_depth, self.min_samples_split)
    self.dTree = root
    return
  
  # Find the class with the given feature vector(list1)
  def findClass(self, btree, list1):
    # If the given feature value is less than threshold, go to the left branch
    if list1[btree['iFeature']] < btree['threshold']:
      # If the branch is a dictionary, recursively search the class
      if isinstance(btree['l_branch'], dict):
        return self.findClass(btree['l_branch'], list1)
      else:
        # If the branch is a class value, return the value
        return btree['l_branch']
    else:
      # If the branch is a dictionary, recursively search the class
      if isinstance(btree['r_branch'], dict):
        return self.findClass(btree['r_branch'], list1)
      else:
        # If the branch is a class value, return the value
        return btree['r_branch']
  
  # Predict the class with the given X matrix
  def predict(self, test):
    pred_list = list()
    for list1 in test.values:
      class_val = self.findClass(self.dTree, list1)
      pred_list.append(class_val)
    return(pred_list)


# Read the Fisher's iris flower dataset
irisData = pd.read_csv('iris.csv', index_col=0)
irisData['species'] = irisData['species'].apply(lambda x: 1 
        if x=='setosa' else 2 if x=='versicolor' else 3)
# X_dat is 150 x 4 matrix
# X_dat has four features: sepal_length, sepal_width, petal_length, petal_width
X_dat = irisData.drop('species', axis=1)
# y_dat is 150 x 1 vector
# y_dat contains the species informaiton
y_dat = irisData['species']

# Split the data into train and test sets. Test set is the 20% of the data.
#dataTrain, dataTest = train_test_split(irisData, test_size = 0.2) 

# Divide the data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X_dat, y_dat, 
              test_size = 0.2, random_state=6)

#model = DecisionTreeClassifier(max_depth=2, min_samples_split=2)
model = DecisionTree(max_depth=2, min_samples_split=2)
model.fit(Xtrain,ytrain)
yPredict = model.predict(Xtest)
score = accuracy_score(ytest, yPredict)
# SVC rbf and xgboost have a very high score
print('The accurancy is %f' % (score))

