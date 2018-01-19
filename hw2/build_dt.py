import sys
import numpy as np
import time
from math import log2
from collections import Counter

class Node:
    def __init__(self, node_path, node_data, node_used_features, node_depth, current_label):
        self.path = node_path
        self.data = node_data
        self.used_features = node_used_features
        self.depth = node_depth
        self.label = current_label
        self.has_child = False
        self.pos_child = None
        self.neg_child = None
        self.current_feature = ''

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


def entropy(label_list, all_labels):
    probs = []
    for l in all_labels:
        probs.append(label_list.count(l))
    return en(probs)


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


def summary(node, all_labels):
    num_samples = len(node.data[0])
    num_clusters = {}
    for l in all_labels:
        num_clusters[l] = (node.data[0] == l).sum()
    print(num_samples)
    for k in num_clusters:
        print(k, num_clusters[k])
    print('-----')

import os
# os.chdir('hw2')


def find_leaf(root, feature):
    if root.has_child:
        if root.current_feature in feature:
            return find_leaf(root.pos_child, feature)
        else:
            return find_leaf(root.neg_child, feature)
    else:
        return root


if __name__ == "__main__":
    use_local_file = True
    if use_local_file:
        training_data_filename = 'examples/train.vectors.txt'
        test_data_filename = 'examples/test.vectors.txt'
        max_depth = 20
        min_gain = 0
        model_file = 'model_file'
        sys_output = 'sys.out'
    else:
        training_data_filename = sys.argv[1]
        test_data_filename = sys.argv[2]
        max_depth = sys.argv[3]
        min_gain = sys.argv[4]
        model_file = sys.argv[5]
        sys_output = sys.argv[6]
    start = time.time()
    # read data training data, data[0] is the list of labels, data[1] is the list of features
    training_data, feature_set = read_data(training_data_filename)
    label_set = set(training_data[0])
    keep_looping = 1
    node_set = []
    root = Node('', training_data, [], 0, '')
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
                        pos_label = Counter(pos_samples[0]).most_common(1)[0][0]
                        pos_path = cur_node.path + best_feature + '&'
                        uf = cur_node.used_features
                        uf.append(best_feature)  # used feature
                        pos_node = Node(pos_path, pos_samples, uf, cur_node.depth+1, pos_label)
                        # negative child
                        neg_path = cur_node.path + '!' + best_feature + '&'
                        neg_label = Counter(neg_samples[0]).most_common(1)[0][0]
                        neg_node = Node(neg_path, neg_samples, uf, cur_node.depth+1, neg_label)
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
        print('looping')
    end = time.time()
    print('total:', (end-start)/60)
    # for item in node_set:
    #     print(item.path)
    # e = entropy(train_labels, set(train_labels))
    # print()
    # for item in training_data:
    #     print(item)
    test_data = read_test_data(test_data_filename)
    yes = 0
    no = 0
    for i in range(len(test_data[0])):
        label = test_data[0][i]
        feature = test_data[1][i]
        leaf = find_leaf(root, feature)
        if leaf.label == label:
            yes += 1
        else:
            no += 1
    print(yes/(yes+no))
    yes = 0
    no = 0
    for i in range(len(training_data[0])):
        label = training_data[0][i]
        feature = training_data[1][i]
        leaf = find_leaf(root, feature)
        if leaf.label == label:
            yes += 1
        else:
            no += 1
    print(yes/(yes+no))

