import numpy as np
import math
import operator
from sklearn import datasets

# Data
iris = datasets.load_iris()
iris_x = iris.data 
iris_y = iris.target

# Shuffle data
randIndex = np.arange(iris_x.shape[0])
np.random.shuffle(randIndex)
iris_x = iris_x[randIndex]
iris_y = iris_y[randIndex]
x_train = iris_x[:100, :] 
x_test = iris_x[100:, :]
y_train = iris_y[:100]
y_test = iris_y[100:]

# KNN algorithm
# Predict label for a given point
def predict(x_train, y_train, point, k):
	neighbour_labels = get_k_neighbours(x_train, y_train, point, k)
	return highest_vote(neighbour_labels)

# Get label of K nearest neighbours
def get_k_neighbours(x_train, y_train, point, k):
	distances = []
	neighbour_labels = []

	for i in range(len(x_train)):
		distance = points_distance(x_train[i], point)
		distances.append((distance, y_train[i]))

	distances.sort(key=operator.itemgetter(0))		# sort by distances

	for i in range(k):
		neighbour_labels.append(distances[i][1])	# get label of k smallest distances

	return neighbour_labels

# Get the most common label
def highest_vote(neighbour_labels):
	labels_count = [0,0,0]

	for label in neighbour_labels:
		labels_count[label] += 1

	max_count = max(labels_count)

	return labels_count.index(max_count)

# Calculate distance between 2 points
def points_distance(p1, p2):
	dimension = len(p1)
	distance = 0
	for i in range(dimension):
		distance += (p1[i]-p2[i])**2
	return math.sqrt(distance)

# Accuracy evaluation
def accuracy_score(predicts, groundTruths):
	total = len(predicts)
	corrects = 0

	for i in range(total):
		if predicts[i] == groundTruths[i]:
			corrects += 1

	return corrects/total

# KNN results
y_predict = []
k = 5
for p in x_test:
	label = predict(x_train, y_train, p, k)
	y_predict.append(label)

accuracy = accuracy_score(y_predict, y_test)
print("Accuracy rate: ", accuracy)