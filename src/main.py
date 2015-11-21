import os
from init import *
from io import *
from feature import *

if __name__ == '__main__':

    #data
    X_train, Y_train = read_csv(train_file_path)
    X_test, _ = read_csv(test_file_path)

    print 'Read {} training examples'.format(len(X_train))
    print 'Read {} testing examples'.format(len(X_test))

    # feature
    #feature_train = extract_feature(X_train)
    #label_digit_train, label_dict = convert_label_digit(Y_train)
    #label_vector_train = convert_label_vector(label_digit_train, label_dict)
    #data_train, label_train = concate_data(feature_train, label_vector_train)

    #learning


    #evaluation
