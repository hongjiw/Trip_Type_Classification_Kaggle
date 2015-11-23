import os
import numpy
from sklearn.preprocessing import normalize

def extract_date(x, _dict):
    dateFeature = {}
    dateDict = _dict
    dateCount = 0
    
    for key in x.keys():
        tmp = x[key][0][0]
        if not dateDict.has_key(tmp):
            dateDict[tmp] = dateCount
            dateCount = dateCount + 1
        dateFeature[key] = dateDict[tmp]
    return dateFeature, dateDict

def extract_department(x, _dict):
    dptFeature = {}
    dptDict = _dict
    dptCount = 0
    
    for key in x.keys():
        dptFeature[key] = {}
        for item in x[key]:
            dptTmp = item[3]
            if not dptDict.has_key(dptTmp):
                dptDict[dptTmp] = dptCount
                dptCount = dptCount + 1
            dptTmpCount = dptDict[dptTmp]
            if not dptFeature[key].has_key(dptTmpCount):
                dptFeature[key][dptTmpCount] = 1
            else:
                dptFeature[key][dptTmpCount] = dptFeature[key][dptTmpCount] + 1
    return dptFeature, dptDict
    
def extract_category(x, _dict):
    categoryFeature = {}
    categoryDict = _dict
    categoryCount = 0
    
    for key in x.keys():
        categoryFeature[key] = {}
        for item in x[key]:
            dptTmp = item[3]
            if not categoryDict.has_key(dptTmp):
                categoryDict[dptTmp] = categoryCount
                categoryCount = categoryCount + 1
            categoryTmpCount = categoryDict[dptTmp]
            if not categoryFeature[key].has_key(categoryTmpCount):
                categoryFeature[key][categoryTmpCount] = 1
            else:
                categoryFeature[key][categoryTmpCount] = categoryFeature[key][categoryTmpCount] + 1
    return categoryFeature, categoryDict

def extract_feature(x, dictlist):
    dateFeature, dateDict = extract_date(x, dictlist[0])
    dptFeature, dptDict = extract_department(x, dictlist[1])
    categoryFeature, categoryDict = extract_category(x, dictlist[2])
    feature = {}
    
    for num in dateFeature.keys():
        tmp = numpy.zeros(len(dateDict), dtype = float)
        tmp[dateFeature[num]] = 1
        tmpfeature = tmp
        
        tmp = numpy.zeros(len(dptDict), dtype = float)
        for key in dptFeature[num].keys():
            tmp[key] = dptFeature[num][key]
        tmp = normalize(tmp.reshape(1, len(dptDict))).reshape(len(dptDict))
        tmpfeature = numpy.hstack((tmpfeature, tmp))
        
        tmp = numpy.zeros(len(categoryDict), dtype = float)
        for key in categoryFeature[num].keys():
            tmp[key] = categoryFeature[num][key]
        tmp = normalize(tmp.reshape(1, len(categoryDict))).reshape(len(categoryDict))
        tmpfeature = numpy.hstack((tmpfeature, tmp))
        feature[num] = tmpfeature
    return feature, [dateDict, dptDict, categoryDict]
    
def convert_label_digit(y):
    labelDict = {}
    labelCount = 0
    labelDigit = {}
    for num in y.keys():
        if not labelDict.has_key(y[num]):
            labelDict[y[num]] = labelCount
            labelCount = labelCount + 1
        labelDigit[num] = labelDict[y[num]]
    return labelDigit, labelDict

def convert_label_vector(y, yDict):
    labelVector = {}
    for num in y.keys():
        labelVector[num] = numpy.zeros(len(yDict))
        labelVector[num][y[num]] = 1
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

















