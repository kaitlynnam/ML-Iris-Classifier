
from sklearn import datasets

import matplotlib.pyplot as plt

iris = datasets.load_iris()

features = iris.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

scatter = plt.scatter(sepal_length,sepal_width, c=iris.target)
 
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)

handles = scatter.legend_elements()[0]
labels = iris.target_names.tolist()  

plt.legend(handles=handles, labels=labels, title="Species")

plt.savefig('iris_scatter.png')

plt.show()