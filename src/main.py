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
    feature, trainDicts = extract_feature(X_train, [{}, {}, {}])
    labelDigit, labelDict = convert_label_digit(Y_train)
    labelVec = convert_label_vector(labelDigit, labelDict)
    X, y_digit, y_vector = concate_data(feature, labelDigit, labelVec)
    
    clf = get_clf()
    # score = k_fold_validate(clf, X, y_digit, y_vector, 5)
    
    feature_test, testDicts = extract_feature(X_test, trainDicts)
    clf.fit(X, y_digit)
    prob_test = clf.predict_proba(feature_test.values())
    write_csv(feature_test.keys(), prob_test, labelDict)

    #split train and val
    # cutoff = int(len(feature) *  (train_val_ratio - 1) / train_val_ratio)
    # feature_train = feature[0:cutoff]
    # label_train = label[0:cutoff]

    # feature_val = feature[cutoff:]
    # label_val = label[cutoff:]

    # print 'Extracted {} training samples'.format(len(feature_train))
    # print 'Extracted {} validation samples'.format(len(feature_val))

    ##learning
    # print 'Training'
    # clf = fit(feature_train, label_train)
    # result = classify(feature_val, clf)

    ##evaluation
    # acc = compute_accuracy(result.tolist(), label_val)
    # print 'Validation Accuracy is {}'.format(acc)
