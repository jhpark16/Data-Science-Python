'''
Gaussian Mixture Model applied to the Fisher's iris flower dataset.
The iris dataset is grouped using unsupervised learning (Gaussian Mixture Model).
Author: Jungho Park
Date: March 20, 2017
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
# Dimensionality reduction using principal component analysis
from sklearn.decomposition import PCA  

# Read the Fisher's iris flower dataset
irisData = pd.read_csv('iris.csv', index_col=0)
# X_dat is 150 x 4 matrix
# X_dat has four features: sepal_length, sepal_width, petal_length, petal_width
X_dat = irisData.drop('species', axis=1)
# y_dat is 150 x 1 vector
# y_dat contains the species informaiton
y_dat = irisData['species']
# Create a conversion table for the dataset.
# Since it is not possible to predict the group numbers, this converstion 
# table should be adjusted to the result of the unsupervised learning
labels = {'setosa':0, 'versicolor':2, 'virginica':1}
y_num = list(map(lambda i: labels[i],y_dat))
# Initialize the Unsupervised Gaussian Mixture model
model = GaussianMixture(n_components=3, covariance_type='full') 

# Fit the model with the dataset
model.fit(X_dat)

# Try to predict the class label using the dataset
y_fitted = model.predict(X_dat)

irisData['GaussianMixture'] = y_fitted
# Evaluate the accuracy of the test dataset
# The accuracy depends on the random_state. It is between 0.9 and 1.0
print(accuracy_score(y_num, y_fitted))

# Choose two principla components
model = PCA(n_components=2)

# Fit the model with the iris dataset
model.fit(X_dat)

# Convert the 4-dimensional iris dataset to 2-D dataset
X_dat_2D = model.transform(X_dat)

# Add the Reduced 2D dataset to the dataframe for plotting
irisData['X_Red1'] = X_dat_2D[:, 0]
irisData['X_Red2'] = X_dat_2D[:, 1]
# Plot the reduced dataset of the dataframe using seaborn package 
sns.lmplot('X_Red1','X_Red2',data=irisData, fit_reg=False, hue='species', 
           col='GaussianMixture', palette="Set1", scatter_kws={'s':50})

