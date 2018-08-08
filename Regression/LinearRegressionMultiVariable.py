'''
Python multi-variable linear regression
Author: Jungho Park
Date: February 12, 2017
'''
import numpy as np
import matplotlib.pyplot as plt

# Load generated data with some random jitter
Data_Multi = np.loadtxt('LinearMultiVar.dat')
ind_vec=np.argsort(Data_Multi[:,0],axis=0)
Data_Multi = Data_Multi[ind_vec,:]
nLen = Data_Multi.shape[0]
# Add a constant term
X_lin_mul = np.c_[np.ones(nLen),Data_Multi[:,:-1]]
y_lin = Data_Multi[:,-1]

# Load generated data with some random jitter
Data_Poly = np.loadtxt('Polynomial.dat')
ind_vec=np.argsort(Data_Poly[:,0],axis=0)
Data_Poly = Data_Poly[ind_vec,:]
nLen_Poly = Data_Poly.shape[0]
# Add a constant term
X_poly = np.c_[np.ones(nLen),Data_Poly[:,:-1]]
y_poly = Data_Poly[:,-1]

# Solve linear equation for linear regression
# y = w0 + w1*x1 + w2*x2 .......
X = X_lin_mul.copy()
y = y_lin.copy()
# Calculate weight vector
w_vector = np.linalg.solve(X.T.dot(X), X.T.dot(y))
# Calculate fitted Y vales
y_fitted = X.dot(w_vector)
# Calculate correlation coefficient
y_mean = np.mean(y)
SS_residual = np.sum((y - y_fitted)**2.0)
SS_total = np.sum((y - y_mean)**2.0)
R = np.sqrt(1.0 - SS_residual/SS_total)
print("R (correlation coefficient) = %.4f" % (R))

# Plot the difference btween y and y_fitted
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,1],X[:,2],y_fitted-y)
plt.show()

# Solve linear equation for polynomial regression
# y = w0 + w1*x^1 + w2*x^2 + w2*x^3 .......
X = X_poly.copy()
y = y_poly.copy()
# Solve for the weight vector using the following equation
# y = x_vector * w_vector (single equation)
# x_vector.T*y = (x_vector.T*x_vector) * w_vector (simultaneous linear equations)
w_vector = np.linalg.solve(X.T.dot(X), X.T.dot(y))
# Calculate fitted Y vales
y_fitted = X.dot(w_vector)
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
