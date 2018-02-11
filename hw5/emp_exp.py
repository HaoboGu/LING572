import sys
import numpy as np
import re
import os


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


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        if 'hw5' in os.listdir():
            os.chdir('hw5')
        training_data_filename = 'examples/train2.vectors.txt'
        sys_output = 'emp_count'
    else:
        training_data_filename = sys.argv[1]
        sys_output = sys.argv[2]

    training_data, feature_set, label_set = read_training_data(training_data_filename)
    n_instance = len(training_data)
    emp_expect = {}
    raw_count = {}
    # Count empirical expectation
    for label, features in training_data:
        # for every training instance
        for feat in features:
            # for every (label, feature) pair
            if (label, feat) not in emp_expect:
                emp_expect[(label, feat)] = 1/n_instance
                raw_count[(label, feat)] = 1
            else:
                emp_expect[(label, feat)] += 1/n_instance
                raw_count[(label, feat)] += 1
    feature_list = sorted(list(feature_set))
    label_list = sorted(list(label_set))
    # Write empirical expectations to the file
    with open(sys_output, 'w') as output_file:
        for label in label_list:
            for feat in feature_list:
                if (label, feat) in emp_expect:
                    output_string = label + ' ' + feat+' ' + ('%.5f ' % emp_expect[(label, feat)]) \
                                    + str(raw_count[(label, feat)]) + '\n'
                    output_file.write(output_string)

                else:
                    output_string = label + ' ' + feat + ' ' + str(0) + ' ' + str(0) + '\n'
                    output_file.write(output_string)


