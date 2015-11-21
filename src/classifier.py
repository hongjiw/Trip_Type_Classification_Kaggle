from sklearn import svm
import numpy as np

def train(X_train, Y_train):
    svm_clf = svm.LinearSVC()
    svm_clf.fit(X_train.values(), y_train.values())

    return svm_clf;

def classify(X_test, svm_clf):
    results = []

    for instance_id, feat in X_test.iteritems():
        label = svm_clf.predict(feat)
        results.append((instance_id, label[0]))

    return results
