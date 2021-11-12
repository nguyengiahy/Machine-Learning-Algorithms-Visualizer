import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
digit = datasets.load_digits()
digit_x = digit.data 
digit_y = digit.target

# Shuffle data
randIndex = np.arange(len(digit_x))
np.random.shuffle(randIndex)
digit_x = digit_x[randIndex]
digit_y = digit_y[randIndex]
x_train, x_test, y_train, y_test = train_test_split(digit_x, digit_y, test_size=360)

# KNN algorithm
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)

# Accuracy result
accuracy = accuracy_score(y_predict, y_test)
print(accuracy)

# Test an example
# Dislay the example
img = x_test[0].reshape(8,8)
plt.gray()
plt.imshow(img)
plt.show()
# Test
print(knn.predict(x_test[0].reshape(1,-1)))
