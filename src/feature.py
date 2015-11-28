import os
import numpy
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

def classify_by_label(labelDigit):
    sampleSet = {}
    for i in range(len(labelDigit)):
        if not sampleSet.has_key(labelDigit[i]):
            sampleSet[labelDigit[i]] = []
        sampleSet[labelDigit[i]].append(i)
    return sampleSet

def correlation(x, sampleSet):
    entropy = 0.0
    sumFreq = x.sum()
    for subset in sampleSet.values():
        p = float(x[subset].sum()) / sumFreq
        entropy = entropy - p * numpy.log(p)
    return entropy

def extract_date(x, _dict):
    '''
    x: [
        [
         ["TripType","VisitNumber","Weekday","Upc","ScanCount","DepartmentDescription","FinelineNumber"]
         ...
        ] 
       ]
    '''
    dateFeature = numpy.zeros(len(x), dtype = int)
    dateDict = _dict
    dateCount = 0
    
    for i in range(len(x)):
        tmp = x[i][0][0] #only use the first record
        if not dateDict.has_key(tmp):
            dateDict[tmp] = dateCount
            dateCount = dateCount + 1
        dateFeature[i] = dateDict[tmp]
    feature = numpy.zeros((len(x), len(dateDict)), dtype = float)
    
    for i in range(len(x)):
        feature[i][dateFeature[i]] = 1
    return feature, dateDict

def extract_department(x, _dict):
    '''
    '''
    dptFeature = []
    dptDict = _dict
    dptCount = 0
    
    for i in range(len(x)):
        dptFeature.append({})
        for item in x[i]:
            dptTmp = item[3]
            if not dptDict.has_key(dptTmp):
                dptDict[dptTmp] = dptCount
                dptCount = dptCount + 1
            dptTmpCount = dptDict[dptTmp]
            if not dptFeature[i].has_key(dptTmpCount):
                dptFeature[i][dptTmpCount] = 1
            else:
                dptFeature[i][dptTmpCount] = dptFeature[i][dptTmpCount] + 1
    feature = numpy.zeros((len(x), len(dptDict)), dtype = float)
    for i in range(len(x)):
        for key in dptFeature[i].keys():
            feature[i][key] = dptFeature[i][key]
    
    return feature, dptDict
    
def extract_category(x, _dict):
    categoryFeature = []
    categoryDict = _dict
    categoryCount = 0
    
    for i in range(len(x)):
        categoryFeature.append({})
        for item in x[i]:
            categoryTmp = item[4]
            if not categoryDict.has_key(categoryTmp):
                categoryDict[categoryTmp] = categoryCount
                categoryCount = categoryCount + 1
            categoryTmpCount = categoryDict[categoryTmp]
            if not categoryFeature[i].has_key(categoryTmpCount):
                categoryFeature[i][categoryTmpCount] = 1
            else:
                categoryFeature[i][categoryTmpCount] = categoryFeature[i][categoryTmpCount] + 1
    feature = numpy.zeros((len(x), len(categoryDict)), dtype = float)
    for i in range(len(x)):
        for key in categoryFeature[i].keys():
            feature[i][key] = categoryFeature[i][key]
    return feature, categoryDict

def extract_department_item_num(x, _dict):
    dptFeature = []
    dptDict = _dict
    
    for i in range(len(x)):
        dptFeature.append({})
        for item in x[i]:
            dptTmp = item[3]
            dptItemNum = int(item[2])
            dptTmpCount = dptDict[dptTmp]
            if not dptFeature[i].has_key(dptTmpCount):
                dptFeature[i][dptTmpCount] = dptItemNum
            else:
                dptFeature[i][dptTmpCount] = dptFeature[i][dptTmpCount] + dptItemNum
    feature = numpy.zeros((len(x), len(dptDict)), dtype = float)
    for i in range(len(x)):
        for key in dptFeature[i].keys():
            feature[i][key] = dptFeature[i][key]
    return feature
    
def extract_feature(x, dictlist):
    dateFeature, dateDict = extract_date(x, dictlist[0])
    
    dptFeature, dptDict = extract_department(x, dictlist[1])
    dptFeatureCount = extract_department_item_num(x, dptDict)
    
    categoryFeature, categoryDict = extract_category(x, dictlist[2])
    
    feature = numpy.hstack((dateFeature, dptFeature))
    feature = numpy.hstack((feature, dptFeatureCount))
    feature = numpy.hstack((feature, categoryFeature))
    
    return feature, [dateDict, dptDict, categoryDict]
    
def convert_label_digit(y):
    labelDict = {}
    labelCount = 0
    labelDigit = numpy.zeros(len(y), dtype = int)
    for i in range(len(y)):
        if not labelDict.has_key(y[i]):
            labelDict[y[i]] = labelCount
            labelCount = labelCount + 1
        labelDigit[i] = labelDict[y[i]]
    return labelDigit, labelDict

def convert_label_vector(y, yDict):
    labelVector = numpy.zeros((len(y), len(yDict)), dtype = int)
    for i in range(len(y)):
        labelVector[i][y[i]] = 1
    return labelVector
        
def concate_digit(x, y):
    data = numpy.zeros((len(x), x.values()[0].shape[0]))
    label = numpy.zeros(len(y))
    i = 0
    for num in x.keys():
        data[i] = x[num]
        label[i] = y[num]
        i = i + 1
    return data, label
    
def concate_vector(x, y):
    data = numpy.zeros((len(x), x.values()[0].shape[0]))
    label = numpy.zeros((len(y), y.values()[0].shape[0]))
    i = 0
    for num in x.keys():
        data[i] = x[num]
        label[i] = y[num]
        i = i + 1
    return data, label

def concate_data(x, y_digit, y_vec):
    data = numpy.zeros((len(x), x.values()[0].shape[0]))
    label_digit = numpy.zeros(len(y_digit))
    label_vec = numpy.zeros((len(y_vec), y_vec.values()[0].shape[0]))
    i = 0
    for num in x.keys():
        data[i] = x[num]
        label_digit[i] = y_digit[num]
        label_vec[i] = y_vec[num]
        i = i + 1
    return data, label_digit, label_vec

















