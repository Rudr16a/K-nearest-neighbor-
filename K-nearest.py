#knn from sklearn 
from sklearn import neigbhors, datasets

#import some data
iris= datasets.load_iris()

X=iris.data[:,:2]
y=iris.target 

clf= neigbhors.KNeighborsClassifiers(n_neighbors=15)
clf.fit(X,y)