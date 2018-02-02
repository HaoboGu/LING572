import sys
import numpy as np
from scipy.spatial.distance import cdist
import os
from collections import Counter
from operator import itemgetter


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
        feature_list.append(feature_d)
        line = f.readline().strip('\n').strip(' ')

    for i in range(len(feature_list)):
        for item in all_features:
            if item not in feature_list[i]:
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


def knn(train, test, k, distance_func):
    """
    Knn algorithm.
    :param train: np.array([label, feature_dict]]
    :param test: np.array([label, feature_dict]]
    :param k: int
    :param distance_func: 1 means using euclidean distance in knn, 2 means cosine distance
    :type distance_func: int
    :return:
    """
    # create lists used in distance calculation
    training_labels = train[0]
    training_lists = train[1]
    test_labels = test[0]
    test_lists = test[1]
    label_set = set(training_labels)
    # calculate distance matrix
    if distance_func == 1:
        matrix = cdist(np.array(test_lists), np.array(training_lists), metric='euclidean')
    elif distance_func == 2:
        matrix = cdist(np.array(test_lists), np.array(training_lists), metric='cosine')
    else:
        return

    result_label = []
    out_strings = []

    for row in range(matrix.shape[0]):
        # for each test instance
        best_labels = training_labels[np.argpartition(matrix[row], k)[:k]]  # get top k similar labels
        counts = dict(Counter(best_labels))  # count them
        for l in label_set:  # add labels that aren't in top k
            if l not in counts:
                counts[l] = 0
        denominator = sum(counts.values())
        sorted_items = sorted(counts.items(), key=itemgetter(1), reverse=True)  # sort results by count
        sorted_items = np.array(sorted_items)
        # rearrange result by prob and then by label name
        max_c = sorted_items[0][1]
        max_s = sorted_items[0][0]
        for index in range(len(sorted_items)):
            c = sorted_items[index][1]
            lb = sorted_items[index][0]
            if max_c == c and lb < max_s:
                sorted_items[index][0], sorted_items[0][0] = sorted_items[0][0], sorted_items[index][0]
                sorted_items[index][1], sorted_items[0][1] = sorted_items[0][1], sorted_items[index][1]
                max_s = sorted_items[0][0]
            elif max_c == c:
                continue
            else:
                break

        result_label.append(sorted_items[0][0])  # add best label to result

        out_string = 'array:' + str(row) + ' ' + test_labels[row]
        for key, value in sorted_items:
            out_string += ' ' + str(key) + ' ' + ("%.5f" % (float(value)/denominator))
        out_string += '\n'
        out_strings.append(out_string)

    return out_strings, result_label


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
    print_str += '\n Training accuracy=' + ("%.5f" % (yes_train/len(train_result))) + '\n'
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

    print_str = ''
    for i in range(len(label_set)):
        print_str += ' ' + map_to_l[i]  # first row
    print_str += '\n'
    for i in range(len(label_set)):
        print_str += map_to_l[i]  # first col
        for j in range(len(label_set)):
            print_str += ' ' + str(int(matrix[i][j]))
        print_str += '\n'
    print_str += '\n testing accuracy=' + ("%.5f" % (yes_test/len(test_result)))+ '\n'
    print(print_str)


def write_system_output(training_result, test_result, output_filename):
    f = open(output_filename, 'w')
    f.write("%%%%% training data:\n")
    for o_str in training_result[0]:
        f.write(o_str)
    f.write("\n\n%%%%% test data:\n")
    for o_str in test_result[0]:
        f.write(o_str)
    f.close()


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        if 'hw4' in os.listdir():
            os.chdir('hw4')
        training_data_filename = 'examples/train.vectors.txt'
        test_data_filename = 'examples/test.vectors.txt'
        k_val = 5
        similarity_func = 2
        sys_output = 'sys.out'
    else:
        training_data_filename = sys.argv[1]
        test_data_filename = sys.argv[2]
        k_val = int(sys.argv[3])
        similarity_func = int(sys.argv[4])
        sys_output = sys.argv[5]

    training_data, word_set = read_data(training_data_filename)
    test_data = read_test_data(test_data_filename, word_set)
    word_list = list(word_set)

    # rearrange training data based on order in word list
    train_instances = []
    train_labels = []
    for label, d in training_data:
        train_list = []
        for i in range(len(word_list)):
            train_list.append(d[word_list[i]])
        train_instances.append(train_list)
        train_labels.append(label)
    train_labels = np.array(train_labels)
    # rearrange test data based on order in word list
    test_instances = []
    test_truth = []
    for label, d in test_data:
        test_list = []
        for i in range(len(word_list)):
            test_list.append(d[word_list[i]])
        test_instances.append(test_list)
        test_truth.append(label)

    # run KNN on test and training data
    result_test = knn([train_labels, train_instances], [test_truth, test_instances], k_val, similarity_func)
    result_train = knn([train_labels, train_instances], [train_labels, train_instances], k_val, similarity_func)

    # Write test results to sys_output
    write_system_output(result_train, result_test, sys_output)
    # Print confusion matrix to command line
    print_confusion_matrix(result_train[1], train_labels, result_test[1], test_truth)






