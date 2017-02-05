import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

clusters = y_train.astype(str)
np.place(clusters, clusters=='0', 'g')
np.place(clusters, clusters=='1', 'r')
np.place(clusters, clusters=='2', 'b')
names = iris_dataset['feature_names']
data1 = iris_dataset['data'][:,0]
data2 = iris_dataset['data'][:,1]
data3 = iris_dataset['data'][:,2]
data4 = iris_dataset['data'][:,3]
groups = ('setosa', 'versicolor', 'virginica')

plt.figure(1)

ax = plt.subplot(441)
ax.hist(data1,color='green',alpha=0.8)

plt.subplot(442)
plt.scatter(data1,data2, c=clusters,  alpha=0.9,label=groups )
plt.title(names[1]+names[0])
plt.ylabel(names[1])
plt.xlabel(names[0])
plt.legend(loc=2)

plt.subplot(443)
plt.scatter(data1,data3, c=clusters,  alpha=0.9,label=groups )
plt.title(names[2]+names[0])
plt.ylabel(names[2])
plt.xlabel(names[0])
plt.legend(loc=2)

plt.subplot(444)
plt.scatter(data1,data4, c=clusters,  alpha=0.9,label=groups )
plt.title(names[3]+names[0])
plt.ylabel(names[3])
plt.xlabel(names[0])
plt.legend(loc=2)

####

plt.subplot(445)
plt.scatter(data2,data1, c=clusters,  alpha=0.9,label=groups )
plt.title(names[0]+names[1])
plt.ylabel(names[0])
plt.xlabel(names[1])
plt.legend(loc=2)

ax = plt.subplot(446)
ax.hist(data2,color='green',alpha=0.8)

plt.subplot(447)
plt.scatter(data2,data3, c=clusters,  alpha=0.9,label=groups )
plt.title(names[2]+names[1])
plt.ylabel(names[2])
plt.xlabel(names[1])
plt.legend(loc=2)

plt.subplot(448)
plt.scatter(data2,data4, c=clusters,  alpha=0.9,label=groups )
plt.title(names[3]+names[1])
plt.ylabel(names[3])
plt.xlabel(names[1])
plt.legend(loc=2)

###
plt.subplot(449)
plt.scatter(data3,data1, c=clusters,  alpha=0.9,label=groups )
plt.title(names[0]+names[2])
plt.ylabel(names[0])
plt.xlabel(names[2])
plt.legend(loc=2)

plt.subplot(4,4,10)
plt.scatter(data3,data2, c=clusters,  alpha=0.9,label=groups )
plt.title(names[3]+names[2])
plt.ylabel(names[3])
plt.xlabel(names[2])
plt.legend(loc=2)

ax = plt.subplot(4,4,11)
ax.hist(data3,color='green',alpha=0.8)

plt.subplot(4,4,12)
plt.scatter(data3,data4, c=clusters,  alpha=0.9,label=groups )
plt.title(names[3]+names[1])
plt.ylabel(names[3])
plt.xlabel(names[1])
plt.legend(loc=2)

##
plt.subplot(4,4,13)
plt.scatter(data4,data1, c=clusters,  alpha=0.9,label=groups )
plt.title(names[0]+names[1])
plt.ylabel(names[0])
plt.xlabel(names[1])
plt.legend(loc=2)


plt.subplot(4,4,14)
plt.scatter(data4,data2, c=clusters,  alpha=0.9,label=groups )
plt.title(names[2]+names[1])
plt.ylabel(names[2])
plt.xlabel(names[1])
plt.legend(loc=2)

plt.subplot(4,4,15)
plt.scatter(data4,data3, c=clusters,  alpha=0.9,label=groups )
plt.title(names[3]+names[1])
plt.ylabel(names[3])
plt.xlabel(names[1])
plt.legend(loc=2)

ax = plt.subplot(4,4,16)
ax.hist(data4,color='green',alpha=0.8)

plt.show()
