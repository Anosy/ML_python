import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus

# 仍然使用自带的iris数据
iris = datasets.load_iris()
# print(iris)
X = iris.data   # 选择0，2种特征的特征值
# print(X)
y = iris.target
clf = DecisionTreeClassifier(criterion='entropy',max_depth=4)
clf.fit(X,y)
def plot_scatter():
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.show()
def plot_tree():   # 绘制决策树
    dot_data = tree.export_graphviz(clf, out_file=None,
                             feature_names=iris.feature_names,
                             class_names=iris.target_names,
                             filled=True, rounded=True,
                             special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('iris1.png')
plot_tree()