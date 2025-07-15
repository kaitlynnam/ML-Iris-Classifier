from sklearn import datasets
from sklearn import tree

import matplotlib.pyplot as plt

iris = datasets.load_iris()

X, y = iris.data, iris.target

clf = tree.DecisionTreeClassifier(random_state=0, max_depth=10, min_samples_leaf=1, criterion='gini', min_samples_split= 5, max_features='log2')
clf = clf.fit(X, y)

tree.plot_tree(clf)
plt.show()

print(clf.predict([[2,2,2,2]]))

"""print(clf.predict_proba([[2,2,2,2]]))"""

# Regression targets need floating point values and can't be used on the iris dataset because it needs continous quantifiable values


"""from sklearn import tree
import matplotlib.pyplot as plt

X = [[0,0],[1,1]]
y = [0.5, 1.5]

clf = tree.DecisionTreeRegressor(random_state=0)
clf = clf.fit(X, y)

tree.plot_tree(clf)
plt.show()"""


'''import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Step 1: Training data (dots)
X = [[1,2], [6,0], [3,6], [1,5], [6,7], [4,5]]
y = [[1.0, 5.3], [7.6, 1.9], [2.3, 9.4], [3.2, 7.3], [4.1, 8.7], [1.1, 9.9]]

# Step 2: Train the model
model = DecisionTreeRegressor()
model.fit(X, y)

# Step 3: Make predictions across smooth range
X_test = np.linspace(0, 3, 100).reshape(-1, 2)
y_pred = model.predict(X_test)

# Step 4: Plot
plt.scatter(X, y, color="red", label="Original data")       # red dots
plt.plot(X_test, y_pred, color="blue", label="Prediction")  # blue line
plt.xlabel("Input (X)")
plt.ylabel("Output (y)")
plt.title("Regression Prediction")
plt.legend()
plt.show()

'''