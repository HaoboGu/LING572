import sys
import numpy as np
from scipy.spatial.distance import cdist
import time
import os


def read_data(filename):
    """
    Read training data
    :param filename:
    :return: np.array([label, feature])
    """
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    all_features = set()
    while line:
        feature_d = {}
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        feature_tuples = []
        for fea in features:
            feature_d[fea.split(':')[0]] = int(fea.split(':')[1])
            all_features.add(fea.split(':')[0])
            feature_tuples.append([fea.split(':')[0], int(fea.split(':')[1])])
        label_list.append(label)
        # feature_list.append(feature_tuples)
        feature_list.append(feature_d)
        line = f.readline().strip('\n').strip(' ')

    for i in range(len(feature_list)):
        # f_set = set(list(map(list, zip(*feature_list[i])))[0])
        for item in all_features:
            if item not in feature_list[i]:
                # feature_list[i].append([item, 0])
                feature_list[i][item] = 0
    data = np.array([label_list, feature_list])
    return data.transpose(), all_features


def read_test_data(filename, features_set):
    """
    Read test data
    :param filename: str
    :param features_set: set
    :return: np.array([label, feature])
    """
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    while line:
        feature_d = {}
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        for fea in features:
            if fea.split(':')[0] in features_set:
                feature_d[fea.split(':')[0]] = int(fea.split(':')[1])
        label_list.append(label)
        feature_list.append(feature_d)
        line = f.readline().strip('\n').strip(' ')
    for item in features_set:
        for i in range(len(feature_list)):
            if item not in feature_list[i]:
                feature_list[i][item] = 0
    data = np.array([label_list, feature_list])
    return data.transpose()


if __name__ == "__main__":
    use_local_file = True
    if use_local_file:
        if 'hw4' in os.listdir():
            os.chdir('hw4')
        training_data_filename = 'examples/train.vectors.txt'
        test_data_filename = 'examples/test.vectors.txt'
        k_val = 5
        similarity_func = 1
        sys_output = 'sys.out'
    else:
        training_data_filename = sys.argv[1]
        test_data_filename = sys.argv[2]
        k_val = int(sys.argv[3])
        similarity_func = int(sys.argv[4])
        sys_output = sys.argv[5]

    s = time.time()
    training_data, word_set = read_data(training_data_filename)
    test_data = read_test_data(test_data_filename, word_set)
    word_list = list(word_set)
    training_lists = []
    training_labels = []
    train_word_list = []
    for train_label, train_d in training_data:
        train_list = []
        for i in range(len(word_list)):
            train_list.append(train_d[word_list[i]])
            train_word_list.append(word_list[i])
        training_lists.append(train_list)
        training_labels.append(train_label)


    test_lists = []
    test_labels = []
    test_word_list = []
    for test_label, test_d in test_data:
        test_list = []
        for i in range(len(word_list)):
            test_list.append(test_d[word_list[i]])
            test_word_list.append(word_list[i])
        test_lists.append(test_list)
        test_labels.append(test_label)
    training_labels = np.array(training_labels)
    correct, wrong = 0, 0
    train_matrix = cdist(np.array(training_lists), np.array(training_lists), metric='euclidean')
    for i in range(train_matrix.shape[0]):
        best_labels = training_labels[np.argpartition(train_matrix[i], k_val)[:k_val]]
        unique, pos = np.unique(best_labels, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        if training_labels[i] == unique[maxpos]:
            correct += 1
        else:
            wrong += 1

    matrix = cdist(np.array(test_lists), np.array(training_lists), metric='cosine')

    correct, wrong = 0, 0
    for i in range(matrix.shape[0]):
        best_labels = training_labels[np.argpartition(matrix[i], k_val)[:k_val]]
        unique, pos = np.unique(best_labels, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        if training_labels[i] == unique[maxpos]:
            correct += 1
        else:
            wrong += 1
    print(wrong, correct)





