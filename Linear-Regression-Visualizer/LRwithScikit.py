import numpy as np  
import matplotlib.pyplot as plt 
from sklearn import linear_model

# Random data
A = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T
plt.plot(A,b,'ro')

# Create model
lr = linear_model.LinearRegression()

# Train model
lr.fit(A,b)

# Draw line
# y=ax+b, a: coefficient, b:intercept
x0 = np.array([[1,46]]).T
y0 = x0 * lr.coef_ + lr.intercept_

# Visualize data
plt.plot(x0, y0)
plt.show()