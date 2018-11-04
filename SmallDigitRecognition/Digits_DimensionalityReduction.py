# -*- coding: utf-8 -*-
'''
Dimensionality Reduction of Small Digit dataset
The dimension was reduced from 64 dimensions to 2 dimensions.
linear (PCA) and non-linear (Isomap) dimenstionality reductions were compared.
Author: Jungho Park
Date: March 1, 2017
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
# Dimensionality reduction using principal component analysis
from sklearn.decomposition import PCA  
from sklearn.manifold import Isomap

# Read the small handwritten digit dataset
digitsData = pd.read_csv('digits_small.csv')
y_digits = digitsData['0']
x_digits = digitsData.drop('0',axis=1)

# Non-linear dimensionality reduction using Isomap
iso = Isomap(n_components=2)
# Fit the isomap model to the digit dataset
iso.fit(x_digits)
# Transform the 64 dimnensional digit dataset to 2 dimnensional dataset
X_dat_2D = iso.transform(x_digits)
# Plot the 2D data
plt.scatter(X_dat_2D[:, 0], X_dat_2D[:, 1], c=y_digits,
            edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);
plt.show()

# Linear dimensionality reudction using PCA
model = PCA(n_components=2)

# Fit the model with the iris dataset
model.fit(x_digits)

# Convert the 4-dimensional iris dataset to 2-D dataset
X_dat_2D2 = model.transform(x_digits)

#Plot the 2D data
plt.scatter(X_dat_2D2[:, 0], X_dat_2D2[:, 1], c=y_digits,
            edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);
plt.show()
