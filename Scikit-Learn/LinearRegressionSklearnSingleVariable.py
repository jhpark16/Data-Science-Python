'''
Python single-variable linear regression using scikit-learn
Author: Jungho Park
Date: February 12, 2017
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
# Change the X vector into a matrix
X = X[:, np.newaxis]
y = y_lin.copy()
# Create a Sklearn LinearRegression model
model = LinearRegression(fit_intercept=True)
model.fit(X,y)
alpha, beta = model.intercept_, model.coef_[0]
# Evaluate the fitted curve
Y_fitted = model.predict(X)
# Plot the data and fitted line
plt.scatter(X,y)
plt.plot(X,Y_fitted)
plt.show()
# Calculate the correlation coefficient
R2 = model.score(X,y)
print("R2 = %.4f" % (np.sqrt(R2)))

# Linear regression applied to an exponential curve
# y = alpha*exp(x*beta)
# ln y = ln(alpha) + beta*x
X = X_exp.copy()
X = X[:, np.newaxis]
#ind = np.argsort(X,axis=0)
y = np.log(y_exp.copy())
# Calculate alpha and beta values
model = LinearRegression(fit_intercept=True)
model.fit(X,y)
alpha, beta = model.intercept_, model.coef_[0]
# Evaluate the fitted curve
Y_fitted = model.predict(X)
# Plot the data and fitted line
plt.scatter(X,y_exp)
plt.plot(X[ind], np.exp(Y_fitted[ind]))
plt.show()
# Calculate the correlation coefficient
R2 = model.score(X,y)
print("R2 = %.4f" % (np.sqrt(R2)))

# Linear regression applied to a power curve
# y = alpha*power(x,beta)
# ln y = ln(alpha) + beta*ln(x)
X = np.log(X_pow.copy())
X = X[:, np.newaxis]
y = np.log(y_pow.copy())
# Calculate alpha and beta values
model = LinearRegression(fit_intercept=True)
model.fit(X,y)
alpha, beta = model.intercept_, model.coef_[0]
# Evaluate the fitted curve
Y_fitted = model.predict(X)
# Plot the data and fitted line
plt.scatter(X_pow,y_pow)
plt.plot(np.exp(X), np.exp(Y_fitted))
plt.show()
# Calculate the correlation coefficient
R2 = model.score(X,y)
print("R2 = %.4f" % (np.sqrt(R2)))

# Linear regression applied to ln linear case
# y = alpha + beta*ln(x)
X = np.log(X_log.copy())
# Obtain sorted index for the data
ind = np.argsort(X_log)
# Apply the index to sort the dataset
X = X[ind, np.newaxis]
y = y_log[ind].copy()
# Calculate alpha and beta values
model = LinearRegression(fit_intercept=True)
model.fit(X,y)
alpha, beta = model.intercept_, model.coef_[0]
# Evaluate the fitted curve
Y_fitted = model.predict(X)
# Plot the data and fitted line
plt.scatter(X_log,y_log)
plt.plot(np.exp(X), Y_fitted)
plt.show()
# Calculate the correlation coefficient
R2 = model.score(X,y)
print("R2 = %.4f" % (np.sqrt(R2)))
