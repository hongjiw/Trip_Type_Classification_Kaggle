import os
from myInit import *
from myIO import *
from feature import *


# data
X_train, Y_train = read_csv(train_file_path)
X_test, _ = read_csv(test_file_path)

print 'Read {} training examples'.format(len(X_train))
print 'Read {} testing examples'.format(len(X_test))

# feature
feature_train = extract_feature(X_train)
label_digit_train, label_dict = convert_label_digit(Y_train)
label_vector_train = convert_label_vector(label_digit_train, label_dict)
# using vector label
data_train, label_train = concate_data(feature_train, label_vector_train)


# learning



# evaluation
