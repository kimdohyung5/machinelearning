

# first get data .
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from my_knn import KNN
import numpy as np

#cmap_lst = [plt.cm.rainbow, plt.cm.Blues, plt.cm.autumn, plt.cm.RdYlGn]
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


data = load_iris()
#print("dir(data)", dir(data))
(data_X, data_y) = (data.data, data.target)
#print("data_X.shape", data_X.shape)

(train_X, test_X, train_y, test_y) = train_test_split( data.data, data.target, test_size=0.2, random_state=42 )

#plt.scatter( train_X[:,0], train_X[:, 1], c=train_y , cmap = cmap)
#plt.show()

knn = KNN(7)
knn.fit(train_X,train_y)
prediction = knn.predict(test_X)
print("prediction", prediction)
print("test_y", test_y)

correct = np.sum( np.equal( prediction , test_y ) )
total = len( test_y )
print(f'accuracy = { 1.0 * correct / total :.4f} ')


