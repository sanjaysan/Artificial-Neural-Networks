from __future__ import division
import pandas as pd
import numpy as np
import scipy.io.arff as arff
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import sys
# import time
import warnings

def initialize_weights(num_input_units, num_hidden_units):
    if num_hidden_units == 0:
        theta1 = np.random.uniform(-0.01, 0.01, (1, num_input_units + 1))
    else:
        theta1 = np.random.uniform(-0.01, 0.01, (num_hidden_units, num_input_units + 1))
    theta2 = np.random.uniform(-0.01, 0.01, (1, num_hidden_units + 1))
    return theta1, theta2

def process_arff_file(file_name):
    loaded_arff = arff.loadarff(open(file_name, "rb"))
    (data_frame, metadata) = loaded_arff
    features = metadata.names()

    data_frame = pd.DataFrame(data_frame)
    data_frame_class = data_frame['class']
    data_frame_class = data_frame_class.apply(lambda class_val: 0 if (class_val == metadata['class'][1][0]) else 1)
    data_frame_without_class = data_frame.drop('class', axis = 1)
    return data_frame_without_class, data_frame_class.values, len(data_frame_without_class), features, metadata

def sigmoid(val):
    return 1 / (1 + np.exp(-val))

def cost_function(theta1, theta2, X, y, input_weight_bias, hidden_weight_bias):
    m = X.shape[0]
    X = np.hstack((input_weight_bias, X))

    theta1_gradient = np.zeros(theta1.shape)
    theta2_gradient = np.zeros(theta2.shape)

    dot_product = np.dot(theta1, X.transpose())
    oj = sigmoid(dot_product)

    oj = np.vstack((hidden_weight_bias, oj))
    z = np.dot(theta2, oj)
    a = sigmoid(z)

    Y = y

    term1 = sum(np.multiply(-Y, np.log(a)))
    term2 = sum(np.multiply((1 - Y), np.log(1 - a)))

    j = sum(term1 - term2) / m

    # Gradient for hidden unit
    delta3 = a - Y
    theta2_gradient = (theta2_gradient + np.dot(delta3, oj.transpose())) / m

    activation = np.multiply(oj, (1 - oj))
    theta2_dot = np.dot(theta2.transpose(), delta3)
    delta2 = np.multiply(theta2_dot, activation)

    theta2_dot = np.dot(delta2, X)
    theta2_dot = np.delete(theta2_dot, 0, 0)

    theta1_gradient = (theta1_gradient + theta2_dot) / m
    return j, theta1_gradient, theta2_gradient

def neural_network_cost_function(theta1, X, y):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    theta1_gradient = np.zeros(theta1.shape)

    z = np.dot(theta1, X.transpose())
    a = sigmoid(z)
    Y = y

    t1 = sum(np.multiply(-Y, np.log(a)))
    t2 = sum(np.multiply((1 - Y), np.log(1 - a)))

    j = sum(t1 - t2) / m

    delta3 = a - Y
    theta1_gradient = (theta1_gradient + np.dot(delta3, X)) / m

    return j, theta1_gradient

def one_of_k_encoding(data_frame, metadata, features):
    for feature in features:
        if metadata[feature][0] == 'nominal' and not feature == 'class':
            encoded_data_frame = pd.get_dummies(data_frame[feature])
            data_frame = data_frame.drop(feature, axis = 1)
            data_frame = pd.concat([data_frame, encoded_data_frame], axis = 1)
    return data_frame

def encode_scale_data(train_data_x, test_data_x, metadata, features):
    train_data_size = len(train_data_x)
    concatenated_data_frame = pd.concat([train_data_x, test_data_x])
    encoded_data_frame = one_of_k_encoding(concatenated_data_frame, metadata, features)

    train_data_x = encoded_data_frame[:train_data_size]
    test_data_x = encoded_data_frame[train_data_size:len(encoded_data_frame)]
    return preprocessing.scale(train_data_x.values), preprocessing.scale(test_data_x.values)

def print_output(activation, predicted_class, test_data_class_labels, class_dict):
    test_data_size = len(test_data_class_labels)
    num_correctly_classified, num_incorrectly_classified = 0, 0
    result = zip(predicted_class, test_data_class_labels)
    for idx, tuple in enumerate(result):
        if class_dict[tuple[0]] == class_dict[tuple[1]]:
            num_correctly_classified += 1
        else:
            num_incorrectly_classified += 1
        print("Activation : {0:.16f}\tPredicted class : {1}\tActual class : {2}".format(activation[idx], class_dict[tuple[0]], class_dict[tuple[1]]))
    correctness = accuracy_score(predicted_class, test_data_class_labels)
    print("Number of correctly classified instances : {0}\tNumber of incorrectly classified instances : {1}"
            .format(str(round(correctness * test_data_size)).rstrip(".0"), str(test_data_size - round(correctness * test_data_size)).rstrip(".0")))
    # print "Number of correctly classified instances : ", num_correctly_classified,"\tNumber of incorrectly classified instances : ", num_incorrectly_classified
    # total = num_correctly_classified + num_incorrectly_classified
    # print num_correctly_classified / total,"\t", num_incorrectly_classified / total

def predict(theta, X, input_weight_bias, hidden_weight_bias):
    if len(theta) == 1:
        theta1 = theta[0]
        theta2 = None
    else:
        theta1 = theta[0]
        theta2 = theta[1]

    m = X.shape[0]
    X = np.hstack((np.repeat(input_weight_bias, m).reshape(m, 1), X))

    dot_product = np.dot(X, theta1.transpose())
    oj = sigmoid(dot_product)

    if theta2 is None:
        a = oj.flatten()
    else:
        oj = np.hstack((np.repeat(hidden_weight_bias, m).reshape(m, 1), oj))
        z = np.dot(oj, theta2.transpose())
        a = sigmoid(z).transpose()[0]

    threshold = np.vectorize(lambda x: 1 if (x > 0.5) else 0)
    return a, threshold(a)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    learning_rate = float(sys.argv[1])
    num_hidden_units = int(sys.argv[2])
    num_training_epochs = int(sys.argv[3])
    train_data_without_class, train_data_class_labels, train_data_size, features, metadata = process_arff_file(sys.argv[4])
    test_data_without_class, test_data_class_labels, test_data_size, _, _ =  process_arff_file(sys.argv[5])
    class_dict = {
        0: metadata['class'][1][0],
        1: metadata['class'][1][1],
    }

    train_data_without_class, test_data_without_class = encode_scale_data(train_data_without_class, test_data_without_class, metadata, features)
    num_input_units = train_data_without_class.shape[1]

    theta1, theta2 = initialize_weights(num_input_units, num_hidden_units)
    input_weight_bias = np.random.uniform(-0.01, 0.01, (1, 1))
    hidden_weight_bias = np.random.uniform(-0.01, 0.01, (1, 1))

    train_data_without_class, train_data_class_labels = shuffle(train_data_without_class, train_data_class_labels)
    j = 0
    for x in xrange(num_training_epochs):
        ce = 0
        for index in xrange(train_data_size):
            if num_hidden_units == 0:
                [j, theta1_gradient] = neural_network_cost_function(theta1, train_data_without_class[index:index + 1], train_data_class_labels[index])
                theta1 -= (learning_rate * theta1_gradient)
                ce += j
            else:
                [j, theta1_gradient, theta2_gradient] = cost_function(theta1, theta2, train_data_without_class[index:index + 1], train_data_class_labels[index], input_weight_bias, hidden_weight_bias)
                theta1 -= (learning_rate * theta1_gradient)
                theta2 -= (learning_rate * theta2_gradient)
                ce += j
        if num_hidden_units == 0:
            theta = [theta1]
        else:
            theta = [theta1, theta2]

        activation, predicted_class = predict(theta, train_data_without_class, input_weight_bias, hidden_weight_bias)
        correctness = accuracy_score(predicted_class, train_data_class_labels)
        print(
            "{0}\t{1}\t{2}\t{3}"
                .format(x + 1, ce, str(round(correctness * train_data_size)).rstrip(".0"), str(train_data_size - round(correctness * train_data_size)).rstrip(".0")))

        activation, predicted_class = predict(theta, test_data_without_class, input_weight_bias, hidden_weight_bias)
        print_output(activation, predicted_class, test_data_class_labels, class_dict)