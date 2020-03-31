
import numpy as np

class LogisticRegression:
	def __init__(self, iters, learning_rate = 0.001):
		self.W = None
		self.b = None
		self.iters = iters
		self.learning_rate = learning_rate

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.W = np.zeros( n_features )
		self.b = 0
		for _ in range( self.iters ):
			linear_preds = np.dot( X, self.W ) + self.b 
			logis = self._sigmoid(linear_preds)
			dw = 1./ n_samples * np.dot( X.T, logis - y )
			db = 1./ n_samples * np.sum( logis - y )
			self.W -= self.learning_rate * dw
			self.b -= self.learning_rate * db 
		
	def predict(self, X):
		linear_preds = np.dot( X, self.W ) + self.b 
		logis = self._sigmoid(linear_preds)
		logis_cls = [ 1 if x > 0.5 else 0 for x in logis]
		return logis_cls
		#return np.array( logis_cls )
		#return logis
		
	def _sigmoid(self, x):
		return 1. / (1. + np.exp(-x))