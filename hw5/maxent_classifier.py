import sys
import numpy as np
import re
from operator import itemgetter
from math import exp


def read_test_data(filename):
    """
    Read test data
    :param filename: str
    :return: np.array([label, feature])
    """
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    while line:
        line = re.sub('\s+', ' ', line)
        feature_d = {}
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        for fea in features:
            feature_d[fea.split(':')[0]] = int(fea.split(':')[1])
        label_list.append(label)
        feature_list.append(feature_d)
        line = f.readline().strip('\n').strip(' ')
    data = np.array([label_list, feature_list])
    return data.transpose()


def read_model(filename):
    """
    Read MaxEnt model from model filename
    :param filename:
    :return: model dictionary and label set
    """
    with open(filename, 'r') as model_f:
        model = {}
        labels = set()
        line = model_f.readline().strip('\n').strip()
        words = line.split(' ')
        if words[0] == "FEATURES":
            current_tag = words[3]  # read first tag
            labels.add(current_tag)
        else:
            return
        line = model_f.readline().strip('\n').strip() # start iterate from second line
        while line:
            words = line.split(' ')
            if words[0] == "FEATURES":
                current_tag = words[3]
                labels.add(current_tag)
            else:
                model[(current_tag, words[0])] = float(words[1])
            line = model_f.readline().strip('\n').strip()

        return model, labels


def run_maxent(model, label_set, data):
    """
    Run maxent decoder on data.
    :param model:
    :param label_set:
    :param data:
    :return: output string and predicted labels
    """
    results = []
    output_strings = []
    i = 0  # index of array
    for label, features in data:
        probs = {}
        # calculate P(y|x) for y=cur_label
        for cur_label in label_set:
            exponent = 0
            for feat in features:
                if (cur_label, feat) in model:
                    exponent += model[(cur_label, feat)]  # sum all elements together
            exponent += model[(cur_label, '<default>')]  # plus default feature
            probs[cur_label] = exp(exponent)
        denominator = sum(probs.values())
        for cur_label in probs:
            probs[cur_label] = probs[cur_label] / denominator  # calculate final probability

        sorted_items = sorted(probs.items(), key=itemgetter(1), reverse=True)  # sort results by probability

        results.append(sorted_items[0][0])  # add predict label to result list

        # construct output string
        output_string = 'array:' + str(i) + ' ' + label
        for key, value in sorted_items:
            output_string += ' ' + str(key) + ' ' + ("%.5f" % value)
        output_string += '\n'
        output_strings.append(output_string)
        i += 1  # increase index
    return output_strings, results


def print_confusion_matrix(test_result, test_truth):
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


def write_system_output(output_string, output_filename):
    f = open(output_filename, 'w')
    f.write("%%%%% test data:\n")
    for o_str in output_string:
        f.write(o_str)
    f.close()


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        test_data_filename = 'examples/test2.vectors.txt'
        model_filename = 'q1/m1.txt'
        sys_output = 'sys.out'
    else:
        test_data_filename = sys.argv[1]
        model_filename = sys.argv[2]
        sys_output = sys.argv[3]

    maxent_model, labels = read_model(model_filename)
    test_data = read_test_data(test_data_filename)
    o_strings, predicted_labels = run_maxent(maxent_model, labels, test_data)
    write_system_output(o_strings, sys_output)
    print_confusion_matrix(predicted_labels, test_data.transpose()[0])







