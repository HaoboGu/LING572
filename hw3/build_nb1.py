import sys
import numpy as np
from math import log2
from collections import Counter
from math import log10


def read_data(filename):
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    all_features = set()
    while line:
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        features = set([f.split(':')[0] for f in features])
        all_features = all_features.union(features)
        label_list.append(label)
        feature_list.append(features)
        # data.append([label, features])
        line = f.readline().strip('\n').strip(' ')
    data = np.array([label_list, feature_list])
    return data, all_features


def read_test_data(filename):
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    while line:
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        features = set([f.split(':')[0] for f in features])
        label_list.append(label)
        feature_list.append(features)
        line = f.readline().strip('\n').strip(' ')
    data = np.array([label_list, feature_list])
    return data


def print_confusion_matrix(train_result, train_truth, test_result, test_truth):
    print('Confusion matrix for the training data:\nrow is the truth, column is the system output\n')
    label_set = list(set(train_truth))
    dimension = len(label_set)
    matrix = np.zeros([dimension, dimension])
    map_to_num = {}
    map_to_l = {}
    index = 0
    yes_train = 0
    for item in label_set:
        map_to_num[item] = index
        map_to_l[index] = item
        index += 1
    for i in range(len(train_truth)):
        if train_result[i] == train_truth[i]:
            col = map_to_num[train_result[i]]
            row = map_to_num[train_truth[i]]
            matrix[row][col] += 1
            yes_train += 1
        else:
            col = map_to_num[train_result[i]]
            row = map_to_num[train_truth[i]]
            matrix[row][col] += 1

    print_str = '            '
    for i in range(len(label_set)):
        print_str += ' ' + map_to_l[i]  # first row
    print_str += '\n'
    for i in range(len(label_set)):
        print_str += map_to_l[i]  # first col
        for j in range(len(label_set)):
            print_str += ' ' + str(int(matrix[i][j]))
        print_str += '\n'
    print_str += '\n Training accuracy=' + str(yes_train/len(train_result)) + '\n'
    print(print_str)

    # print test result
    print('Confusion matrix for the test data:\nrow is the truth, column is the system output\n')
    label_set = list(set(test_truth))
    dimension = len(label_set)
    matrix = np.zeros([dimension, dimension])
    map_to_num = {}
    map_to_l = {}
    index = 0
    yes_test = 0
    for item in label_set:
        map_to_num[item] = index
        map_to_l[index] = item
        index += 1
    for i in range(len(test_truth)):
        if test_result[i] == test_truth[i]:
            col = map_to_num[test_result[i]]
            row = map_to_num[test_truth[i]]
            matrix[row][col] += 1
            yes_test += 1
        else:
            col = map_to_num[test_result[i]]
            row = map_to_num[test_truth[i]]
            matrix[row][col] += 1

    print_str = '            '
    for i in range(len(label_set)):
        print_str += ' ' + map_to_l[i]  # first row
    print_str += '\n'
    for i in range(len(label_set)):
        print_str += map_to_l[i]  # first col
        for j in range(len(label_set)):
            print_str += ' ' + str(int(matrix[i][j]))
        print_str += '\n'
    print_str += '\n testing accuracy=' + str(yes_test/len(test_result)) + '\n'
    print(print_str)


def write_system_output(dt_root, training_data, test_data, output_filename):
    f = open(output_filename, 'w')
    f.write("%%%%% training data:\n")

    f.write("\n\n%%%%% test data:\n")
    f.close()
    # print_confusion_matrix(train_result, train_truth, test_result ,test_truth)


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        training_data_filename = 'examples/train.vectors.txt'
        test_data_filename = 'examples/test.vectors.txt'
        class_prior_delta = 1
        cond_prob_delta = 0.1
        model_file = 'model_file'
        sys_output = 'sys.out'
    else:
        training_data_filename = sys.argv[1]
        test_data_filename = sys.argv[2]
        class_prior_delta = float(sys.argv[3])
        cond_prob_delta = float(sys.argv[4])
        model_file = sys.argv[5]
        sys_output = sys.argv[6]

    # read data training data, data[0] is the list of labels, data[1] is the list of features
    training_data, feature_set = read_data(training_data_filename)
