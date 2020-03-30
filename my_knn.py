import numpy as np
from collections import Counter

class KNN:
	def __init__(self, k):
		self.K = k
	
	def fit(self, train_X, train_y):
		self.train_X = train_X
		self.train_y = train_y
		
	def predict(self, test_x):
		arrReturn = []
		for t in test_x:
			distances = [ self._distance(t, train_x) for train_x in self.train_X]
			aa = np.argsort( distances )[:self.K]
			counter = Counter( self.train_y[aa] ).most_common(1)
			arrReturn.append( counter[0][0] )
			#print("counter", counter[0] )
		return arrReturn
	def _distance(self, test_x, train_x):
		return np.sqrt( np.sum( (train_x - test_x) ** 2 , axis=-1) ) 
		