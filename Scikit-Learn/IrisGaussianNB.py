'''
Gaussian Naive Bayes Model applied to the Fisher's iris flower dataset
Author: Jungho Park
Date: March 20, 2017
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

# Divide the data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X_dat, y_dat, random_state=6)

# Initialize the Gaussian Naive Bayes model
model = GaussianNB()                       

# Fit the model with the train dataset
model.fit(Xtrain, ytrain)

# Evaluated the fitted value using the Xtest dataset
y_fitted = model.predict(Xtest)

# Evaluate the accuracy of the test dataset
# The accuracy depends on the random_state. It is between 0.9 and 1.0
print(accuracy_score(ytest, y_fitted))


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
sns.lmplot('X_Red1','X_Red2',data=irisData, fit_reg=False, hue='species')

        