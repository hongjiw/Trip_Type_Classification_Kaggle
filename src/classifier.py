from sklearn import svm
import numpy as np

def fit(X_train, Y_train):
    svm_clf = svm.LinearSVC()
    svm_clf.fit(X_train, Y_train)

    return svm_clf;

def classify(X_test, svm_clf):
    results = []

    results = svm_clf.predict(X_test)

    return results
