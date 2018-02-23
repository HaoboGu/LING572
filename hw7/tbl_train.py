import sys
import numpy as np
import re
import os
from operator import itemgetter


def read_training_data(filename):
    """
    Read training data
    :param filename: str
    :return: np.array([label, feature]), set of features, set of labels, initial tag
    """
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    line = re.sub('\s+', ' ', line)
    initial_tag = line.split(' ')[0]
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    all_features = set()
    while line:
        line = re.sub('\s+', ' ', line)
        feature_d = {}
        tokens = line.split(' ')
        label = tokens[0]
        features_in_line = tokens[1:]
        for fea in features_in_line:
            feature_d[fea.split(':')[0]] = int(fea.split(':')[1])
            all_features.add(fea.split(':')[0])
        label_list.append(label)
        feature_list.append(feature_d)
        line = f.readline().strip('\n').strip(' ')
    data = np.array([label_list, feature_list])
    return data.transpose(), all_features, set(label_list), initial_tag


def initialize(data, initial_tag):
    """
    Initialize data using default label
    :return: [(feat_set, default_label, golden_label)]
    """
    new_data = []
    for golden_label, feat_set in data:
        new_data.append([feat_set, initial_tag, golden_label])
    return new_data


def increase_net_gain(transformation, net_gain):
    """
    If transformation is in the net gain dictionary, then add one to the value.
    Otherwise, initialize the transformation gain using 1
    :param transformation:
    :param net_gain:
    :return:
    """
    if transformation not in net_gain:
        net_gain[transformation] = 1
    else:
        net_gain[transformation] += 1
    return net_gain


def decrease_net_gain(transformation, net_gain):
    """
    If transformation is in the net gain dictionary, decrease net gain for the value
    Otherwise, initialize the transformation gain using 0
    :param transformation:
    :param net_gain:
    :return:
    """
    if transformation not in net_gain:
        net_gain[transformation] = -1
    else:
        net_gain[transformation] -= 1
    return net_gain


def apply_transformation(transformation, data):
    """
    Apply transformation on current data
    :param transformation:
    :param data:
    :return:
    """
    feat = transformation[0]
    from_label = transformation[1]
    to_label = transformation[2]
    for i in range(len(data)):
        # Iterate through all data
        feat_set, cur_label = data[i][0], data[i][1]
        if feat in feat_set and cur_label == from_label:
            data[i][1] = to_label  # apply transformation
    return data


def generate_transformations(data, minimal_gain, labels, initial_tag):
    """
    Generate transformations using training data.
    :param data: training data
    :param minimal_gain: minimal net gain for transformations
    :param labels: set of labels
    :param initial_tag: default label for transformation
    :return:
    """
    data = initialize(data, initial_tag)
    valid_transformations = []
    current_max_gain = minimal_gain + 1  # must have first run
    while current_max_gain >= minimal_gain:
        net_gain = {}
        for feat_set, cur_label, golden_label in data:
            for feat in feat_set:
                # For every feature in instance x, add one to all transformations that lead to the right answer
                # Subtract by one for all trans that start from the right answer
                for to_label in labels:
                    if to_label != cur_label and to_label == golden_label:
                        net_gain = increase_net_gain((feat, cur_label, to_label), net_gain)
                    elif cur_label == golden_label and to_label != cur_label:
                        net_gain = decrease_net_gain((feat, cur_label, to_label), net_gain)

        # Obtain best transformation
        sorted_net_gain = sorted(net_gain.items(), key=itemgetter(1), reverse=True)
        current_max_gain = sorted_net_gain[0][1]
        if current_max_gain >= minimal_gain:
            valid_transformations.append((sorted_net_gain[0]))
            data = apply_transformation(sorted_net_gain[0][0], data)  # apply best trans on data

    return valid_transformations


def write_model(model_file, initial_tag, valid_transformations):
    """
    Write model to model file
    :param model_file:
    :param initial_tag:
    :param valid_transformations: [(transformation, net_gain)], transformation = (feat, from, to)
    :param net_gain:
    :return:
    """
    with open(model_file, 'w') as model_f:
        first_line = initial_tag + '\n'
        model_f.write(first_line)

        for transformation, gain in valid_transformations:
            output_line = transformation[0] + ' ' + transformation[1] + ' ' + transformation[2] + ' ' \
                          + str(gain) + '\n'
            model_f.write(output_line)


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        if 'hw7' in os.listdir():
            os.chdir('hw7')
        training_data_filename = 'examples/train2.txt'
        model_filename = 'model_file'
        min_gain = 1
    else:
        training_data_filename = sys.argv[1]
        model_filename = sys.argv[2]
        try:
            min_gain = int(sys.argv[3])
            if min_gain <= 0:
                print('Error: min_gain is not a positive integer')
                exit(-1)
        except ValueError:
            print('Error: min_gain is not a positive integer')
            exit(-1)

    training_data, feature_set, label_set, default_label = read_training_data(training_data_filename)

    valid_trans = generate_transformations(training_data, min_gain, label_set, default_label)

    write_model(model_filename, default_label, valid_trans)
