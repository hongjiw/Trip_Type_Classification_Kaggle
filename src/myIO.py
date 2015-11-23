import os
import sys
import csv
from myInit import *

def write_csv(number, results, labelDict):
    with open('../data/result.csv', 'wb') as f:
        writer = csv.writer(f)
        for num, prob in zip(number, results):
            row = [int(num)]
            for i in range(3, 45):
                if labelDict.has_key(str(i)):
                    row.append(prob[labelDict[str(i)]])
            row.append(prob[labelDict[str(999)]])
            writer.writerow(row)

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
