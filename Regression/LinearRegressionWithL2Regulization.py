'''
Python single-variable linear regression
Author: Jungho Park
Date: February 12, 2017
'''
import numpy as np
import matplotlib.pyplot as plt

# Load generated data with some random jitter
X, y = np.loadtxt('LinearOutlier.dat')
nLen = X.shape[0]
# Add a constant term
X = np.c_[np.ones(nLen),X]
nTerms = X.shape[1]

# Normal linear regression
# Solve for the weight vector using the following equation
# y = x_vector * w_vector (single equation)
# x_vector.T*y = (x_vector.T*x_vector) * w_vector (simultaneous linear equations)
w_vector1 = np.linalg.solve(X.T.dot(X), X.T.dot(y))
# Calculate fitted Y vales
y_fitted = X.dot(w_vector1)
# Calculate correlation coefficient
y_mean = np.mean(y)
SS_residual = np.sum((y - y_fitted)**2.0)
SS_total = np.sum((y - y_mean)**2.0)
R = np.sqrt(1.0 - SS_residual/SS_total)
print("R (correlation coefficient) = %.4f" % (R))

# Plot the difference btween y and y_fitted
plt.scatter(X[:,1],y)
plt.plot(X[:,1],y_fitted)
plt.show()

# Linear regression with L2 regulization
# Solve for the weight vector using the following equation
# x_vector.T*y = (lambda*w_vector.T + x_vector.T*x_vector) * w_vector (simultaneous linear equations)
L2 = 300.0
L2_term = L2*np.eye(nTerms)
w_vector2 = np.linalg.solve(L2_term + X.T.dot(X), X.T.dot(y))
# Calculate fitted Y vales
y_fitted = X.dot(w_vector2)
# Calculate correlation coefficient
y_mean = np.mean(y)
SS_residual = np.sum((y - y_fitted)**2.0)
SS_total = np.sum((y - y_mean)**2.0)
R = np.sqrt(1.0 - SS_residual/SS_total)
print("R (correlation coefficient) = %.4f" % (R))

# Plot the difference btween y and y_fitted
plt.scatter(X[:,1],y)
plt.plot(X[:,1],y_fitted)
plt.show()

