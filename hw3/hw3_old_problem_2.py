import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt

# Problem 2

# part (a)
delta = 0.025
X = np.arange(-3.0, 5.0, delta)
Y = np.arange(-2.0, 4.0, delta)
X, Y = np.meshgrid(X, Y)
a = mlab.bivariate_normal(X,Y,2.0,1,1,1,0)

plt.figure()
CS = plt.contour(X, Y, a)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()

# part (c)
delta = 0.025
X = np.arange(-10.0, 10.0, delta)
Y = np.arange(-10.0, 10.0, delta)
X, Y = np.meshgrid(X, Y)
a = mlab.bivariate_normal(X,Y,1,2,0,2,1)
b = mlab.bivariate_normal(X,Y,1,2,2,0,1)
c = a - b

plt.figure()
CS = plt.contour(X, Y, c)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()