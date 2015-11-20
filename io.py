import os
import sys
import csv
from init import *

def print_result():
	print 'TODO'

def read_csv(file_path):
	X = {}
	Y = {}

	with open(file_path, 'rb') as csv_file:
		reader = csv.reader(csv_file, delimiter=',', quotechar='"')

		reader.next() #ignore header
		
		for row in reader:
			if len(row) == 6 or len(row) == 7:
				if len(row) == 7: #train
					#Y
					visitNumber = row[1]
					Y[visitNumber] = row[0]
					row = row[1:]

				visitNumber = row[0]

				#X
				if visitNumber not in X:
					X[visitNumber] = [row[1:]]
				else:
					X[visitNumber].append(row[1:])
			else:
				print 'Invalid input file'
				sys.exit(0)
	return X, Y