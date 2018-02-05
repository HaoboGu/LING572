import sys
import numpy as np

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
        test_data_filename = 'examples/test2.vectors.txt'
        model_filename = 'examples/model.txt'
        sys_output = 'sys.out'
    else:
        test_data_filename = sys.argv[1]
        model_filename = sys.argv[2]
        sys_output = sys.argv[3]


