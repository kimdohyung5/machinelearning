
import matplotlib.pyplot as plt


from my_linear_regression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import numpy as np

def mse_loss(y_true, y_predicted):
	return np.mean( (y_true - y_predicted) ** 2 )

data = make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
train_X, test_X, train_y, test_y = train_test_split( data[0], data[1], test_size=0.3, random_state=42)

lr = LinearRegression(train_X.shape[1], 0.01)
lr.fit( train_X, train_y)
predictions = lr.predict(test_X)
loss = mse_loss( test_y, predictions)
print("loss", loss)

'''
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(train_X[:, 0],train_X[:, 1],train_y,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()
'''


plt.figure(figsize=(10,5))
plt.scatter( train_X[:, 0], train_y, color="b", marker="o", s=30)
#plt.scatter( train_X[:, 1], train_y, color="b", marker="o", s=30)
plt.plot( test_X[:,0],predictions ) 
plt.show()


	