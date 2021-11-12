import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
from sklearn import linear_model

# Configure drawing window
fig1 = plt.figure("Gradient Descent Visualizer")
ax = plt.axes(xlim=(-10,60), ylim=(-1,20))

# Data 
A = np.array([[2,9,7,9,11,16,25,23,22,29,29,35,37,40,46]]).T
b = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T
plt.plot(A,b,'ro')

# Line created by Linear Regression formular
lr = linear_model.LinearRegression()
lr.fit(A,b)
x0 = np.array([[1,46]]).T
y0 = x0 * lr.coef_ + lr.intercept_
plt.plot(x0,y0,color="green")

# Line created by Gradient Descent
# f(x) function
def cost(x):
	m = A.shape[0]
	return 0.5/m * np.linalg.norm(A.dot(x) - b, 2)**2

# f'(x) function
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

# Gradient Descent function
def gradientDescent(x_init, learningRate, iteration):
	m = A.shape[0]
	x_list = [x_init]
	for i in range(iteration):
		x_new = x_list[-1] - learningRate * grad(x_list[-1])	# -1 is the last element in list
		if np.linalg.norm(grad(x_new))/len(x_new) < 1.5:		# just a small number -> stop GD
			break
		x_list.append(x_new)
	return x_list

# Function to draw cost per iteration plot
def costPerIterationPlot():
	iter_list = []
	cost_list = []
	for i in range(len(x_list)):
		iter_list.append(i)
		cost_list.append(cost(x_list[i]))
	plt.plot(iter_list, cost_list)
	plt.ylabel("Cost")
	plt.xlabel("Iteration")
	plt.title("Cost per iteration")
	plt.show()

# Add ones to A
ones = np.ones((A.shape[0],1), dtype = np.int8)
A = np.concatenate((A, ones), axis=1)

# Random initial line
x_init = np.array([[1.],[2.]])
y_init = x_init[0][0] * x0 + x_init[1][0]		# y = ax + b
plt.plot(x0, y_init, color="black")
checkGrad(x_init)

# Run gradient descent
iteration = 100
learningRate = 0.0001
x_list = gradientDescent(x_init, learningRate, iteration)

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
plt.title("Gradient Descent Animation (Line)")

# Plot
plt.show()
costPerIterationPlot()