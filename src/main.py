import os
from myInit import *
from myIO import *
from feature import *
from classifier import *
from evaluation import *

if __name__ == '__main__':

    ##data
    X_train, Y_train = read_csv(train_file_path)
    X_test, _ = read_csv(test_file_path)

    print 'Read {} training samples'.format(len(X_train))
    print 'Read {} testing samples'.format(len(X_test))

    ##feature
    feature = extract_feature(X_train).values()
    label = Y_train.values()

    #split train and val
    cutoff = int(len(feature) *  (train_val_ratio - 1) / train_val_ratio)
    feature_train = feature[0:cutoff]
    label_train = label[0:cutoff]

    feature_val = feature[cutoff:]
    label_val = label[cutoff:]

    print 'Extracted {} training samples'.format(len(feature_train))
    print 'Extracted {} validation samples'.format(len(feature_val))

    ##learning
    print 'Training'
    clf = fit(feature_train, label_train)
    result = classify(feature_val, clf)

    ##evaluation
    acc = compute_accuracy(result.tolist(), label_val)
    print 'Validation Accuracy is {}'.format(acc)
