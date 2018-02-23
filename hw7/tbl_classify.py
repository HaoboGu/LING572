import sys
import numpy as np
import re
import os


def read_test_data(filename):
    """
    Read training data
    :param filename: str
    :return: np.array([label, feature]), set of features, set of labels, initial tag
    """
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    line = re.sub('\s+', ' ', line)
    # initial_tag = line.split(' ')[0]
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
    return data.transpose()


def initialize(data, default_label):
    """
    Initialize data using default label
    :return: [(feat_set, default_label, golden_label, applied trans)]
    """
    new_data = []
    for golden_label, feat_set in data:
        new_data.append([feat_set, default_label, golden_label, []])
    return new_data


def read_model(model_file, n):
    """
    Read model for model file.
    :param model_file:
    :param n: read only first n transformations
    :return:
    """

    with open(model_file, 'r') as model_f:
        default_label = model_f.readline().strip('\n')
        line = model_f.readline().strip('\n')
        n_line = 0
        transformations = []
        while line and n_line < n:
            transformation = line.split(' ')
            transformations.append(transformation)  # [feat, from, to, gain]
            line = model_f.readline().strip('\n')
            n_line += 1
    return transformations, default_label


def apply_transformation(transformation, data):
    """
    Apply transformation on current data
    :param transformation:
    :param data: [(feat_set, default_label, golden_label, applied trans)]
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
            data[i][3].append((feat, from_label, to_label))
    return data


def write_sys_out(data, output_filename):
    """
    Write system output
    :param data: [(feat_set, default_label, golden_label, applied trans)]
    :param output_filename:
    :return:
    """
    with open(output_filename, 'w') as output_f:
        for i in range(len(data)):
            out_str = 'array:' + str(i) + ' ' + data[i][2] + ' ' + data[i][1]
            for trans in data[i][3]:
                out_str = out_str + ' ' + trans[0] + ' ' + trans[1] + ' ' + trans[2]
            out_str = out_str + '\n'
            output_f.write(out_str)


def classify(data, transformations, output_filename):
    """
    Apply transformations one by one to get predict result
    :param data:
    :param transformations:
    :param output_filename:
    :return:
    """
    for i in range(len(transformations)):
        # feat, from_label, to_label = transformations[i][0], transformations[i][1], transformations[i][2]
        data = apply_transformation(transformations[i], data)

    write_sys_out(data, output_filename)
    r, w = 0, 0
    for item in data:
        if item[1] == item[2]:
            r += 1
        else:
            w += 1
    print(r/(r+w))


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        if 'hw7' in os.listdir():
            os.chdir('hw7')
        test_data_filename = 'examples/test2.txt'
        model_filename = 'model_file'
        sys_output = 'sys_output.txt'
        n_trans = 1
    else:
        test_data_filename = sys.argv[1]
        model_filename = sys.argv[2]
        sys_output = sys.argv[3]
        n_trans = int(sys.argv[4])

    model, initial_label = read_model(model_filename, n_trans)

    test_data = read_test_data(test_data_filename)

    test_data = initialize(test_data, initial_label)

    classify(test_data, model, sys_output)
