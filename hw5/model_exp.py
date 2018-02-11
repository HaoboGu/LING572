import sys
import numpy as np
import re
import os
from math import exp


def read_training_data(filename):
    """
    Read training data
    :param filename: str
    :return: np.array([label, feature]), set of features, set of labels
    """
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    feature_set = set()
    while line:
        line = re.sub('\s+', ' ', line)
        feature_d = {}
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        for fea in features:
            feature_d[fea.split(':')[0]] = int(fea.split(':')[1])
            feature_set.add(fea.split(':')[0])
        label_list.append(label)
        feature_list.append(feature_d)
        line = f.readline().strip('\n').strip(' ')
    data = np.array([label_list, feature_list])
    return data.transpose(), feature_set, set(label_list)


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
        line = model_f.readline().strip('\n').strip()  # start iterate from second line
        while line:
            words = line.split(' ')
            if words[0] == "FEATURES":
                current_tag = words[3]
                labels.add(current_tag)
            else:
                model[(current_tag, words[0])] = float(words[1])
            line = model_f.readline().strip('\n').strip()

        return model, labels


def read_weights(use_model, model_filename, label_set, feature_set):
    """
    Read p(y|x). If use_model=True, read p(y|x) from model file; else use 1/n_label as p(y|x)
    :param use_model:
    :param model_filename:
    :param label_set:
    :param feature_set:
    :return: dictionary P(label, feature)
    """
    weights = {}
    if use_model:
        weights, label_set = read_model(model_filename)
    else:
        n_label = len(label_set)
        for label in label_set:
            for feat in feature_set:
                weights[(label, feat)] = 1/n_label
            weights[(label, '<default>')] = 1/n_label
    return weights


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        if 'hw5' in os.listdir():
            os.chdir('hw5')
        training_data_filename = 'examples/train2.vectors.txt'
        sys_output = 'model_count'
        model_filename = 'q1/m1.txt'
        use_model = False
    else:
        if len(sys.argv) == 3:
            # no model file given
            training_data_filename = sys.argv[1]
            sys_output = sys.argv[2]
            model_filename = ''
            use_model = False
        else:
            # read model file
            training_data_filename = sys.argv[1]
            sys_output = sys.argv[2]
            model_filename = sys.argv[3]
            use_model = True

    training_data, feature_set, label_set = read_training_data(training_data_filename)
    p = read_weights(use_model, model_filename, label_set, feature_set)
    n_instance = len(training_data)
    model_expect = {}
    raw_count = {}
    # Count model expectation
    for label, features in training_data:
        # Calculate P(y|x) first
        probs = {}
        # for every training instance
        for cur_label in label_set:
            exponent = 0
            for feat in features:
                if (cur_label, feat) in p:
                    exponent += p[(cur_label, feat)]  # sum all elements together
            exponent += p[(cur_label, '<default>')]  # plus default feature
            probs[cur_label] = exp(exponent)
        denominator = sum(probs.values())
        for cur_label in probs:
            probs[cur_label] = probs[cur_label] / denominator  # calculate final probability

        # Then calculate model expectation
        for feat in features:
            for cur_label in label_set:
                if (cur_label, feat) not in model_expect:
                    model_expect[(cur_label, feat)] = probs[cur_label] / n_instance
                    raw_count[(cur_label, feat)] = probs[cur_label]
                else:
                    model_expect[(cur_label, feat)] += probs[cur_label] / n_instance
                    raw_count[(cur_label, feat)] += probs[cur_label]

    feature_list = sorted(list(feature_set))
    label_list = sorted(list(label_set))

    # Write empirical expectations to the file
    with open(sys_output, 'w') as output_file:
        for label in label_list:
            for feat in feature_list:
                if (label, feat) in model_expect:
                    output_string = label + ' ' + feat+' ' + ('%.5f ' % model_expect[(label, feat)]) \
                                    + ('%.5f ' % raw_count[(label, feat)]) + '\n'
                    output_file.write(output_string)


