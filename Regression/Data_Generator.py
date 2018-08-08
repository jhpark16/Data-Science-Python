'''
Python data generator for linear regression
Author: Jungho Park
Date: February 12, 2017
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Number of data points
nData = 100
# Number of variables for multi-variable case
nVar = 3
# Maximum X range of the data
X_min = -10.0
X_max = 10.0
# Random vector generation
X_ran = np.random.rand(nData)
X_ranMulti = np.random.rand(nData,nVar)
# Random jitter generation
jitter = np.random.rand(nData,2)-0.5
jitter_multi = np.random.rand(nData,nVar+1)-0.5
X1 = X_ran*(X_max-X_min)+X_min
X1_multi = X_ranMulti*(X_max-X_min)+X_min
X1.sort()


# Linear data - single variable
# y = alpha+ beta*X
# alpha coefficient
alpha = 1.3
# beta coefficient
beta = 2.8
# The amount of random error
jitter_scale = [1.1, 2.5]
y = alpha+jitter[:,0]*jitter_scale[0]+(beta+jitter[:,1]*jitter_scale[1])*X1
np.savetxt('Linear.dat',(X1,y))
plt.scatter(X1,y)
plt.show()
# Data = np.loadtxt('Linear.dat')

# Linear data with outliers - single variable
# y = alpha+ beta*X
# alpha coefficient
alpha = 1.3
# beta coefficient
beta = 2.8
# The amount of random error
jitter_scale = [1.1, 2.5]
y = alpha+jitter[:,0]*jitter_scale[0]+(beta+jitter[:,1]*jitter_scale[1])*X1
# Add outliers
y[-1] += 200
y[-2] += 200
np.savetxt('LinearOutlier.dat',(X1,y))
plt.scatter(X1,y)
plt.show()
# Data = np.loadtxt('Linear.dat')

# Linear data - multiple variable
# y = alpha + beta1*x1 + beta2*x2
# alpha coefficient
alpha = 1.3
# beta coefficient
beta = [3.6, 1.3]
# The amount of random error
jitter_scale = [0.5, 0.8, 1.1]
y = (alpha + jitter_multi[:,0]*jitter_scale[0] + X1_multi[:,0]*(beta[0]+jitter_multi[:,1]*jitter_scale[1])+
    X1_multi[:,1]*(beta[1]+jitter_multi[:,2]*jitter_scale[2]))
y = y.reshape(nData,1)
X1_m_concat = np.c_[X1_multi,y]
np.savetxt('LinearMultiVar.dat', X1_m_concat)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X1_m_concat[:,0],X1_m_concat[:,1],y)
plt.show()
# Data = np.loadtxt('Linear.dat')

# Linear data - polynomial
# y = alpha + beta1*X + beta2*X^2 + beta3*X^3 + beta4*X^4
# alpha coefficient
alpha = 1.3
# beta coefficient
#beta = [5.6, -2.4, -3.3, -0.2]
beta = [-2.5, 0.4, 0.1]
# The amount of random error
jitter_scale = [10.5, 1.0, 0.2, 0.05]
X1_multi = np.empty((nData,nVar),dtype=float)
y = (alpha + jitter_multi[:,0]*jitter_scale[0])
for i in range(len(beta)):
    X1_multi[:,i] = X1.copy()**(i+1)
    y += X1_multi[:,i]*(beta[i]+jitter_multi[:,i+1]*jitter_scale[i+1])
y = y.reshape(nData,1)
X1_m_concat = np.c_[X1_multi,y]
np.savetxt('Polynomial.dat', X1_m_concat)
plt.scatter(X1_multi[:,0],y)
plt.show()

# Exponential Data - single variable
# y = alpha*exp(x*beta)
# ln y = ln alpha + beta * X
# alpha coefficient
alpha = 2.1 
# beta coefficient
beta = 1.3
# The amount of random error
jitter_scale = [3.0, 0.5]
y = (alpha+jitter[:,0]*jitter_scale[0])*np.exp((beta+jitter[:,1]*jitter_scale[1])*X1)
np.savetxt('Exponential.dat',(X1,y))
plt.scatter(X1,np.log(y))
plt.show()

# Power Data - single variable
# y = alpha*power(x, beta)
# ln y = ln alpha + beta * ln(X)
# alpha coefficient
alpha = 3.1
# beta coefficient
beta = 1.3
# The amount of random error
jitter_scale = [2.0, 0.5]
X2 = X1-X_min
y = (alpha+jitter[:,0]*jitter_scale[0])*np.power(X2,(beta+jitter[:,0]*jitter_scale[1]))
np.savetxt('Power.dat',(X2,y))
plt.scatter(np.log(X2),np.log(y))
plt.show()

# Log Data - single variable
# y = alpha + beta*ln(x)
# alpha coefficient
alpha = 2.1
# beta coefficient
beta = 1.3
# The amount of random error
jitter_scale = [1.3, 0.5]
X2 = X_ran
y = (alpha+jitter[:,0]*jitter_scale[0]) + np.log(X2)*(beta+jitter[:,1]*jitter_scale[1])
np.savetxt('Log.dat',(X2,y))
plt.scatter(np.log(X2),y)
plt.show()

