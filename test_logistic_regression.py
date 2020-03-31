
import numpy as np
from my_logistic_regression import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
X,y = load_breast_cancer(True)

def accuracy(y_true, y_pred):
	accuracy = 1.0 * np.sum(y_true == y_pred) / len(y_true)
	return accuracy

train_X, test_X, train_y, test_y = train_test_split( X, y, test_size=0.2, random_state =1234 )

lr = LogisticRegression(3000)
lr.fit( train_X, train_y)
predictions = lr.predict( test_X )

print(f'LR accuracy ', accuracy(predictions, test_y) )