import numpy as np 
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data
iris = datasets.load_iris()
iris_x = iris.data 
iris_y = iris.target

# Shuffle data
randIndex = np.arange(iris_x.shape[0])
np.random.shuffle(randIndex)
iris_x = iris_x[randIndex]
iris_y = iris_y[randIndex]
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=50)

# KNN algorithm
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)

# Result
accuracy = accuracy_score(y_predict, y_test)
print("Accuracy rate: ", accuracy)