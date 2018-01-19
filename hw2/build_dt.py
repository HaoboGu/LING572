import sys
import numpy as np
from math import log2
from collections import Counter

class Node:
    def __init__(self, node_path, node_data, node_used_features, node_depth, current_label, cur_result):
        self.path = node_path
        self.data = node_data
        self.used_features = node_used_features
        self.depth = node_depth
        self.label = current_label
        self.has_child = False
        self.pos_child = None
        self.neg_child = None
        self.current_feature = ''
        self.result = cur_result


def read_data(filename):
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    all_features = set()
    while line:
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        features = set([f.split(':')[0] for f in features])
        all_features = all_features.union(features)
        label_list.append(label)
        feature_list.append(features)
        # data.append([label, features])
        line = f.readline().strip('\n').strip(' ')
    data = np.array([label_list, feature_list])
    return data, all_features


def read_test_data(filename):
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    while line:
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        features = set([f.split(':')[0] for f in features])
        label_list.append(label)
        feature_list.append(features)
        line = f.readline().strip('\n').strip(' ')
    data = np.array([label_list, feature_list])
    return data


def en(l):
    # compute entropy
    s = sum(l)
    probs = []
    for item in l:
        p = item/s
        probs.append(p)
    s = 0
    for i in probs:
        if i > 0:
            s += i*log2(i)
    return -s


def info_gain(ori_data, pos_data, neg_data, all_labels):
    # ori_data is divided into pos_data and neg_data
    p_ori, p1, p2 = [], [], []
    if pos_data.shape[1] == 0 or neg_data.shape[1] == 0:
        return 0
    for l in all_labels:  # for all possible labels
        p_ori.append((ori_data[0] == l).sum())
        p1.append((pos_data[0] == l).sum())
        p2.append((neg_data[0] == l).sum())
    return en(p_ori) - sum(p1) * en(p1)/sum(p_ori) - sum(p2) * en(p2)/sum(p_ori)


def divide(data, used_feature):
    # divide original data into two parts according to used_feature
    # one part contains all data that has the used_feature
    # another part contains all data that doesn't have used_feature
    ori_label, ori_feature = data[0], data[1]
    pos_indices = [used_feature in item for item in ori_feature]
    neg_indices = [used_feature not in item for item in ori_feature]

    positive_labels = ori_label[pos_indices]
    negative_labels = ori_label[neg_indices]
    pos_features = ori_feature[pos_indices]
    neg_features = ori_feature[neg_indices]
    return np.array([positive_labels, pos_features]), np.array([negative_labels, neg_features])


def find_leaf(node, feature):
    if node.has_child:
        if node.current_feature in feature:
            return find_leaf(node.pos_child, feature)
        else:
            return find_leaf(node.neg_child, feature)
    else:
        return node


def write_dt(root_node, model_filename):
    f = open(model_filename, 'w')
    label_set = set(root_node.data[0])

    def __visit(child, f, label_set):
        if child.has_child:
            __visit(child.pos_child, f, label_set)
            __visit(child.neg_child, f, label_set)
        else:  # leaf node
            out_string = child.path.strip('&') + ' ' + str(child.data.shape[1])
            for l in child.result:
                out_string += ' ' + l + ' ' + str(child.result[l])
            out_string += '\n'
            f.write(out_string)

    if root_node.has_child:
        __visit(root_node.pos_child, f, label_set)
        __visit(root_node.neg_child, f, label_set)
    f.close()


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
    print_str += '\n Training accuracy=' + str(yes_train/len(train_result)) + '\n'
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

    print_str = '            '
    for i in range(len(label_set)):
        print_str += ' ' + map_to_l[i]  # first row
    print_str += '\n'
    for i in range(len(label_set)):
        print_str += map_to_l[i]  # first col
        for j in range(len(label_set)):
            print_str += ' ' + str(int(matrix[i][j]))
        print_str += '\n'
    print_str += '\n testing accuracy=' + str(yes_test/len(test_result)) + '\n'
    print(print_str)


def write_system_output(dt_root, training_data, test_data, output_filename):
    f = open(output_filename, 'w')
    f.write("%%%%% training data:\n")
    train_result = []
    train_truth = []
    # write results of training data
    for i in range(len(training_data[0])):
        label = training_data[0][i]
        feature = training_data[1][i]
        leaf = find_leaf(dt_root, feature)
        out_string = 'array:' + str(i)
        train_result.append(leaf.label)
        train_truth.append(label)
        for key in leaf.result:
            out_string += ' ' + key + ' ' + str(leaf.result[key])
        out_string += '\n'
        f.write(out_string)
    f.write("\n\n%%%%% test data:\n")
    test_result, test_truth = [], []
    # write results of test data
    for i in range(len(test_data[0])):
        label = test_data[0][i]
        feature = test_data[1][i]
        leaf = find_leaf(dt_root, feature)
        out_string = 'array:' + str(i)
        test_result.append(leaf.label)
        test_truth.append(label)
        for key in leaf.result:
            out_string += ' ' + key + ' ' + str(leaf.result[key])
        out_string += '\n'
        f.write(out_string)
    f.close()
    print_confusion_matrix(train_result, train_truth, test_result ,test_truth)

if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        training_data_filename = 'examples/train.vectors.txt'
        test_data_filename = 'examples/test.vectors.txt'
        max_depth = 1
        min_gain = 0.1
        model_file = 'model_file'
        sys_output = 'sys.out'
    else:
        training_data_filename = sys.argv[1]
        test_data_filename = sys.argv[2]
        max_depth = int(sys.argv[3])
        min_gain = float(sys.argv[4])
        model_file = sys.argv[5]
        sys_output = sys.argv[6]

    # read data training data, data[0] is the list of labels, data[1] is the list of features
    training_data, feature_set = read_data(training_data_filename)
    label_set = set(training_data[0])
    keep_looping = 1
    node_set = []
    root = Node('', training_data, [], 0, '', {})
    node_set.append(root)
    useless_path = set()

    while keep_looping:  # there is at least one node is split in previous loop
        keep_looping = 0
        new_node_set = []
        for cur_node in node_set:
            if cur_node.path in useless_path:  #
                new_node_set.append(cur_node)
            else:
                best_feature = ''
                max_ig = 0
                if cur_node.depth < max_depth:  # current_depth < max_depth
                    cur_data = cur_node.data
                    cur_feature_set = feature_set.difference(set(cur_node.used_features))
                    for item in cur_feature_set:  # find best feature
                        pos_samples, neg_samples = divide(cur_data, item)
                        ig = info_gain(cur_data, pos_samples, neg_samples, label_set)
                        if ig > max_ig:
                            best_feature = item
                            max_ig = ig
                    # print('best feature found for', cur_node.path, ':', best_feature, 'with infogain:', max_ig)
                    if max_ig < min_gain or max_ig == 0:  # info_gain is too small, don't split this node
                        cur_node.has_child = False
                        new_node_set.append(cur_node)
                        useless_path.add(cur_node.path)
                        continue
                    else:  # do splitting on this node
                        pos_samples, neg_samples = divide(cur_data, best_feature)
                        # positive child
                        pos_num_instances = pos_samples.shape[1]
                        pos_occurrences = Counter(pos_samples[0])
                        pos_result = {}  # label:probability
                        for l in label_set:
                            p = pos_occurrences.get(l) / pos_num_instances if l in pos_occurrences else 0
                            pos_result[l] = p
                        pos_label = pos_occurrences.most_common(1)[0][0]
                        pos_path = cur_node.path + best_feature + '&'
                        uf = cur_node.used_features
                        uf.append(best_feature)  # used feature
                        pos_node = Node(pos_path, pos_samples, uf, cur_node.depth+1, pos_label, pos_result)
                        # negative child
                        neg_num_instances = neg_samples.shape[1]
                        neg_occurrences = Counter(neg_samples[0])
                        neg_result = {}  # label:probability
                        for l in label_set:
                            p = neg_occurrences.get(l) / neg_num_instances if l in neg_occurrences else 0
                            neg_result[l] = p
                        neg_label = neg_occurrences.most_common(1)[0][0]
                        neg_path = cur_node.path + '!' + best_feature + '&'
                        neg_node = Node(neg_path, neg_samples, uf, cur_node.depth+1, neg_label, neg_result)
                        new_node_set.append(pos_node)
                        new_node_set.append(neg_node)
                        # Update current node
                        cur_node.has_child = True
                        cur_node.current_feature = best_feature
                        cur_node.pos_child = pos_node
                        cur_node.neg_child = neg_node
                        new_node_set.append(cur_node)
                        useless_path.add(cur_node.path)
                        keep_looping = 1
        if keep_looping == 1:
            node_set = new_node_set

    test_data = read_test_data(test_data_filename)
    write_dt(root, model_file)
    write_system_output(root, training_data, test_data, sys_output)

