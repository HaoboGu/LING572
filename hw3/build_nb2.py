import sys
import numpy as np
from math import log2
from collections import Counter
from math import log10
from operator import itemgetter
from decimal import Decimal
import time
from math import factorial as fac


def read_data(filename):
    f = open(filename)
    line = f.readline().strip('\n').strip(' ')
    label_list = []
    feature_list = []  # label_list and feature_list are corresponding to each other
    all_features = set()
    while line:
        feature_d = {}
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        for fea in features:
            feature_d[fea.split(':')[0]] = int(fea.split(':')[1])
        all_features = all_features.union(set(feature_d.keys()))
        label_list.append(label)
        feature_list.append(feature_d)
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
        feature_d = {}
        label = tokens[0]
        features = tokens[1:]
        for fea in features:
            feature_d[fea.split(':')[0]] = int(fea.split(':')[1])
        label_list.append(label)
        feature_list.append(feature_d)
        line = f.readline().strip('\n').strip(' ')
    data = np.array([label_list, feature_list])
    return data


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


def write_system_output(training_result, test_result, output_filename):
    f = open(output_filename, 'w')
    f.write("%%%%% training data:\n")
    for o_str in training_result[0]:
        f.write(o_str)
    f.write("\n\n%%%%% test data:\n")
    for o_str in test_result[0]:
        f.write(o_str)
    f.close()


def save_model(model, label_set, feature_set, model_filename):
    """
    Save Naive Bayes model to model_file
    """
    log_p_wc = model[0]
    p_wc = model[1]
    p_c = model[2]
    model_f = open(model_filename, 'w')
    model_f.write('%%%%% prior prob P(c) %%%%%\n')  # print first line

    # print P(c)
    for l in label_set:
        o_str = l + ' ' + str(p_c[l]) + ' ' + str(log10(p_c[l])) + '\n'
        model_f.write(o_str)

    # print and store logP(w|c), P(w|c) and return them
    model_f.write('%%%%% conditional prob P(f|c) %%%%%\n')
    for l in label_set:
        o_str = '%%%%% conditional prob P(f|c) c=' + l + ' %%%%%\n'
        model_f.write(o_str)
        for f in sorted(feature_set):
            o_str = f + ' ' + l + ' ' + str(p_wc[(l, f)]) + ' ' + str(log_p_wc[(l, f)]) + '\n'
            model_f.write(o_str)
    model_f.close()


def train_model(data, feature_set, class_prior, prob_prior):
    """
    Train the model using multinomial model
    :param data: Training data, [label_list, feature_dictionary]
    :param feature_set: Set of features
    :param class_prior:
    :param prob_prior:
    :return: [logP(w|c), P(w|c), P(c)], label_set]
    """
    label_set = set(data[0])
    n_instance = data.shape[1]
    n_word = len(feature_set)
    # First, count cnt(w,c) and cnt(c) in the corpus
    count_wc = {}
    count_c = {}
    for l in label_set:
        count_c[l] = 0
    for i in range(n_instance):
        cur_data_dict = data[1][i]
        cur_label = data[0][i]
        for word in cur_data_dict:
            if (cur_label, word) not in count_wc:
                count_wc[(cur_label, word)] = cur_data_dict[word]
            else:
                count_wc[(cur_label, word)] += cur_data_dict[word]
            count_c[cur_label] += cur_data_dict[word]

    # Second, compute P(w|c) using cnt(w,c) and cnt(c)
    p_wc = {}
    log_p_wc = {}
    for c in label_set:
        for w in feature_set:
            if (c, w) not in count_wc:
                p_wc[(c, w)] = prob_prior / (n_word*prob_prior + count_c[c])
            else:
                p_wc[(c, w)] = (prob_prior + count_wc[(c, w)]) / (n_word*prob_prior + count_c[c])
            log_p_wc[(c, w)] = log10(p_wc[(c, w)])
    p_c = {}
    for c in label_set:
        p_c[c] = (class_prior + (data[0] == l).sum()) / (len(label_set) * class_prior + n_instance)
    return np.array([log_p_wc, p_wc, p_c]), label_set


def run_test(data, label_set, feature_set, model):
    """
    Run test on data based on given model
    :param data: test data in [label_list, feature_dictionary] form
    :param label_set: set of labels
    :param model: model=[logP(w|c), p(w|c), p(c)]
    :param sys_file: output sys file
    """
    # s = time.time()
    log_p_wc = model[0]  # logP(w|c)
    # p_wc = model[1]  # p(w|c)
    p_c = model[2]  # p(c)
    result = []
    out_strings = []
    correct, wrong = 0, 0

    for i in range(data.shape[1]):  # i is the doc index
        log_p_xc_pc = {}  # log(P(x|c)P(c)), where x is a doc(line)
        cur_word_dict = data[1][i]
        for label in label_set:  # compute log(P(x|ci)P(ci)), iterate through labels
            log_p_xc_pc[label] = log10(p_c[label])
            for w in feature_set:  # calculate log(P(x|ci)P(ci))
                if w in cur_word_dict:
                    log_p_xc_pc[label] = log_p_xc_pc[label] + \
                                      (log_p_wc[(label, w)]*cur_word_dict[w] -
                                       sum([log10(n) for n in range(1, cur_word_dict[w]+1)]))
        result_p = {}
        multiplier = sorted(log_p_xc_pc.values())[1]
        for label in label_set:
            # compute P(ci|x) for each ci: P(ci|x)=P(x|ci)P(ci)/P(x) = P(x|ci)P(ci)/sigma_i P(x|ci)P(ci)
            result_p[label] = float((Decimal(10)**(Decimal(log_p_xc_pc[label]-multiplier))) /
                                    sum([Decimal(10)**Decimal(log_p_xc_pc[l]-multiplier) for l in label_set]))
        sorted_item = sorted(result_p.items(), key=itemgetter(1), reverse=True)  # sort results by probability

        # Construct output string list

        result.append(sorted_item[0][0])  # add predict label to result list
        out_string = 'array:' + str(i) + ' ' + data[0][i]
        for key, value in sorted_item:
            out_string += ' ' + str(key) + ' ' + str(value)
        out_string += '\n'
        out_strings.append(out_string)
    return out_strings, result


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        training_data_filename = 'examples/train.vectors.txt'
        test_data_filename = 'examples/test.vectors.txt'
        class_prior_delta = 0
        cond_prob_delta = 0.1
        model_file = 'model_file'
        sys_output = 'sys.out'
    else:
        training_data_filename = sys.argv[1]
        test_data_filename = sys.argv[2]
        class_prior_delta = float(sys.argv[3])
        cond_prob_delta = float(sys.argv[4])
        model_file = sys.argv[5]
        sys_output = sys.argv[6]

    # read data training data, data[0] is the list of labels, data[1] is the dict of features
    training_data, word_set = read_data(training_data_filename)
    # Training part
    multinomial_model, class_set = train_model(training_data, word_set, class_prior_delta, cond_prob_delta)

    # save model to the model file
    save_model(multinomial_model, class_set, word_set, model_file)

    # Test part
    # Read test data
    test_data = read_test_data(test_data_filename)
    # Test training data using saved model
    training_result = run_test(training_data, class_set, word_set, multinomial_model)
    # Test test data using saved model
    test_result = run_test(test_data, class_set, word_set, multinomial_model)
    # Write test results to sys_output
    write_system_output(training_result, test_result, sys_output)
    # Print confusion matrix to command line
    print_confusion_matrix(training_result[1], training_data[0], test_result[1], test_data[0])



