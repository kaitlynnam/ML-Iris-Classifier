import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

logreg = LogisticRegression(C=1e5)
logreg.fit(X, y)

_, ax = plt.subplots(figsize=(7,4))
DecisionBoundaryDisplay.from_estimator(
    logreg,
    X,
    cmap=plt.cm.Paired,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    xlabel="Sepal length",
    ylabel="Sepal width",
    eps=0.5,
)

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired)

plt.xticks(())
plt.yticks(())
plt.show()