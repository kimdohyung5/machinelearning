
import numpy as np

class LinearRegression:
	def __init__(self, w_len, learning_rate, epochs = 300):
		self.w = np.zeros(w_len)
		self.b = 0
		self.learning_rate = learning_rate
		self.epochs = epochs
		#print("self.w.shape", self.w.shape)
	def fit(self, train_X, train_y):
		instance_len = train_X.shape[0]
		for epoch in range(self.epochs):
			predicted = np.dot( train_X, self.w) + self.b 
			j = predicted - train_y
			#print("train_X.shape", train_X.shape, "j.shape", j.shape)
			dw = (2./ instance_len) * np.dot( train_X.T, j)
			db =  (2./ instance_len) * np.sum(j) 
			#print("dw", dw)
			self.w = self.w - self.learning_rate * dw
			self.b = self.b - self.learning_rate * db
			#if epoch % 2 == 0: print(f'self.epoch={epoch}, self.w={self.w}')
		print(f'w values = {self.w}')
		
	def predict(self, test_X):
		return np.dot( test_X, self.w) + self.b