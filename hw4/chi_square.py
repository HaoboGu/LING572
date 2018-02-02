import sys
import numpy as np
from collections import Counter
from operator import itemgetter
import os


def read_data(use_local_file):
    """
    Read data from local file or from command line
    :param use_local_file: bool
    :return: [label_list, feature_list]
    """
    if use_local_file:
        if 'hw4' in os.listdir():
            os.chdir('hw4')
        f = open('examples/train.vectors.txt')
        line = f.readline().strip('\n').strip(' ')
    else:
        line = sys.stdin.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    all_features = set()
    while line:
        tokens = line.split(' ')
        label = tokens[0]
        line_features = tokens[1:]
        line_feature_set = set()
        for fea in line_features:
            line_feature_set.add(fea.split(':')[0])
            all_features.add(fea.split(':')[0])
        label_list.append(label)
        feature_list.append(line_feature_set)

        if use_local_file:
            line = f.readline().strip('\n').strip(' ')
        else:
            line = sys.stdin.readline().strip('\n').strip(' ')

    if use_local_file:
        f.close()
    return [label_list, feature_list], all_features


def count_f(data, feature, labels):
    """
    Count feature in data
    :param data: [label_list, feature_list]
    :param feature: str
    :param labels: set()
    :return: dictionary with labels as keys
    """
    occurrence = {}
    for l in labels:
        occurrence[l] = 0
    for l, f_set in data:
        if feature in f_set:
            occurrence[l] += 1
    return occurrence


if __name__ == "__main__":
    data, feature_set = read_data(use_local_file=False)
    a = Counter(data[0])
    label_set = set(data[0])
    data = np.array(data).transpose()
    n = sum(a.values())
    result = []
    for feat in feature_set:
        # for every feature
        b = count_f(data, feat, label_set)
        rest_b = {}
        for label in label_set:
            rest_b[label] = a[label] - b[label]
        row_b = sum(b.values())
        row_rest_b = sum(rest_b.values())
        e_b = {}
        e_rest_b = {}
        for label in label_set:
            e_b[label] = a[label] * row_b / n
            e_rest_b[label] = a[label] * row_rest_b / n
        s_b = sum([(b[label]-e_b[label])**2/e_b[label] for label in label_set])
        s_rest_b = sum([(rest_b[label]-e_rest_b[label])**2/e_rest_b[label] for label in label_set])
        chi = s_b + s_rest_b
        result.append([feat, chi, row_b])

    result = sorted(result, key=itemgetter(1), reverse=True)
    for item in result:
        out_string = item[0] + ' %0.5f' % item[1] + ' ' + str(item[2])
        print(out_string)


