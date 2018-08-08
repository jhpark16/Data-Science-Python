'''
Python single-variable linear regression
Author: Jungho Park
Date: February 12, 2017
'''
import numpy as np
import matplotlib.pyplot as plt

def cost(X, y, theta):
    '''
    Simple cost calculation
    '''
    m = y.shape[0]
    return np.sum((X.dot(theta)-y)**2)/(2*m)


def costL1(X, y, theta, L1):
    '''
    Cost_L1 calculates the difference (cost) between X.dot(w_vector)-y
    '''
    m = y.shape[0]
    return np.sum((X.dot(theta)-y)**2)+L1*np.sum(np.abs(theta))/(2*m)

def costL2(X, y, theta, L2):
    '''
    Cost_L1 calculates the difference (cost) between X.dot(w_vector)-y
    '''
    m = y.shape[0]
    return np.sum((X.dot(theta)-y)**2)+L2*np.sum(theta**2)/(2*m)

def costLx(X, y, theta, Lx, xL1):
    '''
    Cost_L1 calculates the difference (cost) between X.dot(w_vector)-y
    '''
    return np.sum((X.dot(theta)-y)**2)+Lx*(xL1*np.sum(np.abs(theta)) + (1-xL1)*np.sum(theta**2))

def gradientDescentSimple(X, y, theta, alpha, tol=1e-8, nIter=10000):
    '''
    Gradient descent algorithm without any regularization
    Return: cost of the final result
    '''
    m = y.shape[0]
    for i in range(nIter):
        # Calculate y estimate using X and weight vector(theta)
        y_calc = X.dot(theta)
        # Calculate the difference between y estimates and y values
        diff = y_calc -y
        # Calculate the correction term for gradient descent
        correction = -alpha/m*(X.T.dot(diff))
#        print(i,np.abs(correction).sum())
#       If the correction is less than tolerance, quit the iteration
        if np.abs(correction).sum()<tol:
            break
        # Modify the theta with the correction values
        theta += correction
    return cost(X, y, theta)

def gradientDescentL1(X, y, theta, alpha, L1, tol=1e-8, nIter=10000):
    '''
    Gradient descent algorithm with L1 regularization
    Return: cost of the final result with L1 regularization
    '''
    m = y.shape[0]
    for i in range(nIter):
        y_calc = X.dot(theta)
        diff = y_calc -y
        correction = -alpha/m*(X.T.dot(diff) + L1/2*np.sign(theta))
#        print(i,np.abs(correction).sum())
        if np.abs(correction).sum()<tol:
            break
        theta += correction
    return costL1(X, y, theta, L1)

def gradientDescentL2(X, y, theta, alpha, L2, tol=1e-8, nIter=10000):
    '''
    Gradient descent algorithm with L2 regularization
    Return: cost of the final result with L2 regularization
    '''
    m = y.shape[0]
    for i in range(nIter):
        y_calc = X.dot(theta)
        diff = y_calc -y
        correction = -alpha/m*(X.T.dot(diff) + L2*theta)
#        print(i,np.abs(correction).sum())
        if np.abs(correction).sum()<tol:
            break
        theta += correction
    return costL2(X, y, theta, L2)

Data_Poly = np.loadtxt('Polynomial.dat')
ind_vec=np.argsort(Data_Poly[:,0],axis=0)
Data_Poly = Data_Poly[ind_vec,:]
nLen = Data_Poly.shape[0]
# Add a constant term
X = np.c_[np.ones(nLen),Data_Poly[:,:-1]]
nTerms = X.shape[1]
y = Data_Poly[:,-1]

# Normal linear regression
# Solve for the weight vector using the following equation
# y = x_vector * w_vector (single equation)
# x_vector.T*y = (x_vector.T*x_vector) * w_vector (simultaneous linear equations)
#w_vector1 = np.linalg.solve(X.T.dot(X), X.T.dot(y))
# alpha is the learning rate
alpha = 1e-5
nSize = X.shape[1]
theta0 = np.random.random(nSize)/np.sqrt(nSize)
theta1 = theta0.copy()
theta2 = theta0.copy()
L1 = 50000
L2 = 50000
# Calculate theta values with the given learning rate (alpha)
loss0 = gradientDescentSimple(X,y,theta0,alpha,nIter=50000)
loss1 = gradientDescentL1(X,y,theta1,alpha,L1,nIter=50000)
loss2 = gradientDescentL2(X,y,theta2,alpha,L2,nIter=50000)
# Calculate fitted Y vales
y_fitted0 = X.dot(theta0)
y_fitted1 = X.dot(theta1)
y_fitted2 = X.dot(theta2)
# Calculate y mean value
y_mean = np.mean(y)
# Calculate residual terms
SS_residual0 = np.sum((y - y_fitted)**2.0)
SS_residual1 = np.sum((y - y_fitted1)**2.0)
SS_residual2 = np.sum((y - y_fitted2)**2.0)
SS_total = np.sum((y - y_mean)**2.0)
# Calculate correlation coefficients
R0 = np.sqrt(1.0 - SS_residual0/SS_total)
R1 = np.sqrt(1.0 - SS_residual1/SS_total)
R2 = np.sqrt(1.0 - SS_residual2/SS_total)
print("R (correlation coefficient without regularization) = %.4f" % (R))
print("R (correlation coefficient with L1(50000) regularization) = %.4f" % (R))
print("R (correlation coefficient with L2(50000) regularization) = %.4f" % (R))

# Plot the difference btween y and y_fitted
plt.scatter(X[:,1],y)
plt.plot(X[:,1],y_fitted0)
plt.plot(X[:,1],y_fitted1)
plt.plot(X[:,1],y_fitted2)
plt.show()

# Linear regression using linear algebra and L2 regulization
# Solve for the weight vector using the following equation
# x_vector.T*y = (lambda*w_vector.T + x_vector.T*x_vector) * w_vector (simultaneous linear equations)
L2 = 50000.0
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

