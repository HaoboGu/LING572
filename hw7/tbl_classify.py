import sys
import numpy as np
import re
import os


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


__name__ = "__main__"
if __name__ == "__main__":
    use_local_file = True
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

