import numpy as np
import matplotlib.pyplot as plt

# Predict label of each datapoint Xi
def predict(w, X):
	'''
	w: result (given).
	X: a Nxd array, each row is a datapoint. Each datapoint has d features.
	label of Xi = 1 if Xi.w > 0 otherwise label = -1
	'''
	return np.sign(X.dot(w))

# Perceptron Learning Algorithm
def perceptron(X, y, w_init):
	'''
	X: a Nxd array, each row is a datapoint. Each datapoint has d features.
	y: 1xN array, label of each row of X. y[i] = 1 or -1.
	w_init: initial random result.
	'''
	w = w_init
	while True:
		pred = predict(w,X)
		# Find indexes of misclassified points
		mis_indxs = np.where(np.equal(pred,y) == False)[0]
		# Number of misclassified points
		num_mis = mis_indxs.shape[0]
		if num_mis == 0:		# no more misclassified points
			return w
		# Random pick 1 misclassified point
		random_id = np.random.choice(mis_indxs, 1)[0]
		# Update w
		w = w + y[random_id] * X[random_id]

# Data
N = 10
X0 = np.array([[4,3,3,4,7,2,5,4,3,3]]).T
Y0 = np.array([[-1,0,1,-2,0,-1,0,1,2,-3]]).T
X1 = np.array([[-2,-1,-2,-5,-3,-1,-4,-4,-3,-4]]).T
Y1 = np.array([[1,-1,-2,-1,3,2,1,-3,-2,-4]]).T
plt.plot(X0,Y0,'ro')
plt.plot(X1,Y1,'bs')
data1 = np.concatenate((X0,Y0), axis=1)
data2 = np.concatenate((X1,Y1), axis=1)
X = np.concatenate((data1, data2), axis=0)
y = np.concatenate((np.ones(N), -1*np.ones(N)))
Xbar = np.concatenate((np.ones((2*N, 1)), X), axis = 1)

# Start PLA
w_init = np.random.randn(Xbar.shape[1])
w = perceptron(Xbar, y, w_init)
line_x = np.array([[-5,5]]).T
line_y = (-w[0] - w[1]*line_x)/w[2]
plt.plot(line_x,line_y,'g')
plt.show()
