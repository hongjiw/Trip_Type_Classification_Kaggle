from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import numpy as np

def get_clf():
    clf = RandomForestClassifier(
        n_estimators = 10, 
        max_depth = None, 
        min_samples_split=1, 
        random_state=0
    )
    
    return clf

def fit(X_train, Y_train):
    svm_clf = svm.LinearSVC()
    svm_clf.fit(X_train, Y_train)

    return svm_clf;

def classify(X_test, svm_clf):
    results = []

    results = svm_clf.predict(X_test)

    return results
