import os
from init import *
from io import *

if __name__ == '__main__':

	#data
	X_train, Y_train = read_csv(train_file_path)
	X_test, _ = read_csv(test_file_path)
	
	print 'Read {} training examples'.format(len(X_train))
	print 'Read {} testing examples'.format(len(X_test))

	#learning



	#evaluation
