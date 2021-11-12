import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
import matplotlib.animation as animation

# Configure drawing window
fig1 = plt.figure("Nesterov Accelerated Gradient")
ax = plt.axes(xlim=(-10,60), ylim=(-1,20))

# Data
A = np.array([[2,9,7,9,11,16,25,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T
plt.plot(A,b,'ro')

# Line created by Linear Regression
lr = linear_model.LinearRegression()
lr.fit(A,b)
x0 = np.array([[1,46]]).T
y0 = lr.coef_ * x0 + lr.intercept_
plt.plot(x0, y0, color="green")

# Line created by GD
# Cost function at model parameters x
def cost(x):
	m = A.shape[0]
	return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2

# Grad function
def grad(x):
	m = A.shape[0]
	return 1/m * A.T.dot(A.dot(x) - b)

# Function to check if f'(x) is correct
def checkGrad(x):
	eps = 1e-4
	g = np.zeros_like(x)
	
	for i in range(len(x)):
		x1 = x.copy()
		x2 = x.copy()
		x1[i] += eps
		x2[i] -= eps
		g[i] = (cost(x1)-cost(x2))/(2*eps)

	g_grad = grad(x)
	if np.linalg.norm(g-g_grad) > 1e-7:
		print("WARNING: CHECK GRADIENT FUNCTION")

# Nesterov Accelerated Gradient
def GD_NAG(x_init, learning_rate, iteration, gamma):
	m = A.shape[0]
	x_list = [x_init]
	v_old = np.zeros_like(x_init)
	for i in range(iteration):
		v_new = gamma*v_old + learningRate*grad(x_list[-1] - gamma*v_old)
		x_new = x_list[-1] - v_new
		if np.linalg.norm(grad(x_new))/len(x_new) < 0.1:
			break
		x_list.append(x_new)
		v_old = v_new
	return x_list

# Add ones to A
ones = np.ones((A.shape[0],1), dtype=np.int8)
A = np.concatenate((A,ones), axis=1)

# Random initial line
x_init = np.array([[1.],[2.]])
y_init = x_init[0][0] * x0 + x_init[1][0]
plt.plot(x0, y_init, color='black')
checkGrad(x_init)

# Run gradient descent
iteration = 1000
learningRate = 0.0001
gamma = 0.9
x_list = GD_NAG(x_init, learningRate, iteration, gamma)

# Draw gradient descent results
for i in range(len(x_list)):
	y0_x_list = x_list[i][0] * x0 + x_list[i][1]
	plt.plot(x0, y0_x_list, color="black", alpha=0.5)

# Draw gradient descent animation
line , = ax.plot([], [], color = "blue")
def update(i):
	y0 = x_list[i][0] * x0 + x_list[i][1]
	line.set_data(x0, y0)
	return line,
iters = np.arange(1,len(x_list),1)
line_animation = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True) 

# Legends
plt.legend(("Data", "Solution by formular", "Initial value for GD", 
			"Value in each GD iteration"), loc=(0.52, 0.01))
legend_text = plt.gca().get_legend().get_texts() 

# Title
plt.title("Nesterov Accelerated Gradient")

# Plot
plt.show()
