import numpy as np
import matplotlib.pyplot as plt

# Random set
A = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
b = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]
plt.plot(A,b,'ro')

# Convert row vector to column vector
A = np.array([A]).T
b = np.array([b]).T

# Create x-square column
x_square = np.array([A[:,0]**2]).T

# Create ones matrix
ones = np.ones((A.shape[0], 1), dtype=np.int8)	# 15x1

# Concatenate
A = np.concatenate((x_square, A, ones), axis = 1)

# Apply linear regression formula
x = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))

# Visualize result
x0 = np.linspace(1,25,10000)
y0 = x[0][0] * (x0**2) + x[1][0] * x0 + x[2][0]
plt.plot(x0, y0)

plt.show()