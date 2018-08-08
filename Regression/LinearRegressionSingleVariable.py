'''
Python single-variable linear regression
Author: Jungho Park
Date: February 12, 2017
'''
import numpy as np
import matplotlib.pyplot as plt

def linear_regression(X, y):
    '''
    Solve for linear regression
    Input: linearized X and y vector
    Return: alpha(offset) and beta(slope) values
    '''
    N = len(X)
    denom = N*X.dot(X)-X.sum()**2.0
    alpha = (y.sum()*X.dot(X)-X.sum()*X.dot(y))/denom
    beta = (N*X.dot(y)-y.sum()*X.sum() )/denom
    return alpha, beta

# Load generated data with some random jitter
X_lin, y_lin = np.loadtxt('Linear.dat')
# Exponential curve data
X_exp, y_exp = np.loadtxt('Exponential.dat')
# Power curve data
X_pow, y_pow = np.loadtxt('Power.dat')
# Log-linear curve data
X_log, y_log = np.loadtxt('Log.dat')


# Solve linear equation for linear regression
# y = alpha + beta*X
X = X_lin.copy()
y = y_lin.copy()
# Calculate alpha and beta values
alpha, beta = linear_regression(X, y)
# Evaluate the fitted curve
Y_fitted = alpha + beta*X
# Plot the data and fitted line
plt.scatter(X,y)
plt.plot(X,Y_fitted)
plt.show()
# Calculate correlation coefficient
y_mean = np.mean(y)
SS_residual = np.sum((y - Y_fitted)**2.0)
SS_total = np.sum((y - y_mean)**2.0)
R = np.sqrt(1.0 - SS_residual/SS_total)
print("R (correlation coefficient) = %.4f" % (R))

# Linear regression applied to an exponential curve
# y = alpha*exp(x*beta)
# ln y = ln(alpha) + beta*x
X = X_exp.copy()
y = np.log(y_exp.copy())
# Calculate alpha and beta values
alpha, beta = linear_regression(X, y)
# Evaluate the fitted curve
Y_fitted = alpha + beta * X
# Plot the data and fitted line
plt.scatter(X,y)
plt.plot(X, Y_fitted)
plt.show()
# Calculate correlation coefficient
y_mean = np.mean(y)
SS_residual = np.sum((y - Y_fitted)**2.0)
SS_total = np.sum((y - y_mean)**2.0)
R = np.sqrt(1.0 - SS_residual/SS_total)
print("R (correlation coefficient) = %.4f" % (R))

# Linear regression applied to a power curve
# y = alpha*power(x,beta)
# ln y = ln(alpha) + beta*ln(x)
X = np.log(X_pow.copy())
y = np.log(y_pow.copy())
# Calculate alpha and beta values
alpha, beta = linear_regression(X, y)
# Evaluate the fitted curve
Y_fitted = alpha + beta * X
# Plot the data and fitted line
plt.scatter(X,y)
plt.plot(X, Y_fitted)
plt.show()
# Calculate correlation coefficient
y_mean = np.mean(y)
SS_residual = np.sum((y - Y_fitted)**2.0)
SS_total = np.sum((y - y_mean)**2.0)
R = np.sqrt(1.0 - SS_residual/SS_total)
print("R (correlation coefficient) = %.4f" % (R))

# Linear regression applied to ln linear case
# y = alpha + beta*ln(x)
X = np.log(X_log.copy())
y = y_log.copy()
# Calculate alpha and beta values
alpha, beta = linear_regression(X, y)
# Evaluate the fitted curve
Y_fitted = alpha + beta * X
# Plot the data and fitted line
plt.scatter(X,y)
plt.plot(X, Y_fitted)
plt.show()
# Calculate correlation coefficient
y_mean = np.mean(y)
SS_residual = np.sum((y - Y_fitted)**2.0)
SS_total = np.sum((y - y_mean)**2.0)
R = np.sqrt(1.0 - SS_residual/SS_total)
print("R (correlation coefficient) = %.4f" % (R))

