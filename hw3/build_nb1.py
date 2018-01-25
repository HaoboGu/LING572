import sys
import numpy as np
from math import log2
from collections import Counter
from math import log10
from operator import itemgetter
from decimal import Decimal
import time


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
        features = set([fea.split(':')[0] for fea in features])
        label_list.append(label)
        feature_list.append(features)
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


def save_model(probs, label_set, feature_set, model_filename):
    """
    Save Naive Bayes model, return model=[log_p_wc, p_wc]
    """
    p_c = probs[0]
    p_wc = probs[1]
    model_f = open(model_filename, 'w')
    model_f.write('%%%%% prior prob P(c) %%%%%\n')  # print first line

    # print P(c)
    for l in label_set:
        o_str = l + ' ' + str(p_c[l]) + ' ' + str(log10(p_c[l])) + '\n'
        model_f.write(o_str)

    # print and store logP(w|c), P(w|c) and return them
    log10_prob_wc = {}
    model_f.write('%%%%% conditional prob P(f|c) %%%%%\n')
    for l in label_set:
        o_str = '%%%%% conditional prob P(f|c) c=' + l + ' %%%%%\n'
        model_f.write(o_str)
        for f in sorted(feature_set):
            log10_prob_wc[(l, f)] = log10(p_wc[(l, f)])
            o_str = f + ' ' + l + ' ' + str(p_wc[(l, f)]) + ' ' + str(log10_prob_wc[(l, f)]) + '\n'
            model_f.write(o_str)
    model_f.close()
    return log10_prob_wc, p_wc, p_c


def training(data, feature_set, class_prior, prob_prior):
    label_set = set(data[0])
    n_instance = data.shape[1]
    p_c = {}  # class prior: p_c = count(c_i)/all_count(c)
    count_c = {}
    for label in label_set:
        count_c[label] = (data[0] == label).sum()
        p_c[label] = (class_prior + count_c[label]) / (len(label_set)*class_prior + n_instance)
    # print(p_c)
    c_wc = {}  # keys are (word, class) tuples, values are count(w,c)
    p_wc = {}  # keys are (feature, class) tuples,  values are probability of 1+count(w,c)/2+count(c)
    adder = len(feature_set)*prob_prior
    for label, words in data.transpose():
        for word in words:
            if (label, word) not in c_wc:
                p_wc[(label, word)] = (prob_prior+1) / (adder + count_c[label])
                c_wc[(label, word)] = 1
            else:
                p_wc[(label, word)] += 1 / (adder + count_c[label])
                c_wc[(label, word)] += 1

    default_p = {}
    for label in label_set:
        default_p[label] = prob_prior / (adder + count_c[label])  # if count(w,c) == 0

    for label in label_set:
        for fea in feature_set:
            if (label, fea) not in p_wc:
                p_wc[(label, fea)] = default_p[label]

    return [p_c, p_wc], label_set


def run_test(data, label_set, model, sys_file):
    """
    Run test on data based on given model
    :param data: test data
    :param label_set: set of labels
    :param model: model=[logP(w|c), p(w|c)]
    :param sys_file: output sys file
    """
    # s = time.time()
    # sys_f = open(sys_file, 'w')
    log_p_wc = model[0]  # logP(w|c)
    p_wc = model[1]  # p(w|c)
    p_c = model[2]  # p(c)
    # correct = 0
    # wrong = 0
    result = []
    out_strings = []
    for i in range(data.shape[1]):
        log_p_xc = {}  # log(P(x|c)), where x is a doc(line)
        cur_words = set(data[1][i])
        for label in label_set:  # compute log(P(x|ci)), iterate through labels
            log_p_xc[label] = 0
            for w in word_set:  # calculate P(x|c) based on page 22 in NB slides
                if w in cur_words:
                    log_p_xc[label] += log_p_wc[(label, w)]
                else:
                    log_p_xc[label] += log10(1 - p_wc[(label, w)])
        result_p = {}
        multiplier = sorted(log_p_xc.values())[0]
        for label in label_set:
            # compute P(ci|x) for each ci: P(ci|x)=P(x|ci)P(ci)/P(x) = P(x|ci)P(ci)/sigma_i P(x|ci)P(ci)
            result_p[label] = \
                float(Decimal(10)**Decimal((log_p_xc[label]-multiplier))*Decimal(p_c[label]) / \
                sum([Decimal(10)**Decimal((log_p_xc[l]-multiplier))*Decimal(p_c[l]) for l in label_set]))
        sorted_item = sorted(result_p.items(), key=itemgetter(1), reverse=True)
        out_string = 'array:' + str(i) + ' ' + data[0][i]
        result.append(sorted_item[0][0])
        # if sorted_item[0][0] == data[0][i]:
        #     correct += 1
        # else:
        #     wrong += 1
        for key, value in sorted_item:
            out_string += ' ' + str(key) + ' ' + str(value)
        out_string += '\n'
        # sys_f.write(out_string)
        out_strings.append(out_string)
    # print(correct, wrong, correct/(correct+wrong))
    # e = time.time()
    # print(e-s)
    # sys_f.close()
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

    # read data training data, data[0] is the list of labels, data[1] is the list of features
    training_data, word_set = read_data(training_data_filename)
    # Training part
    # ps = [p(c), p(w|c)]
    ps, class_set = training(training_data, word_set, class_prior_delta, cond_prob_delta)

    # save model to the model file
    prob_model = save_model(ps, class_set, word_set, model_file)

    # Test part
    test_data = read_test_data(test_data_filename)  # test_data[0] are labels, test_data[1] are words
    # print('start test training data')
    training_result = run_test(training_data, class_set, prob_model, sys_output)
    # print('start test test data')
    test_result = run_test(test_data, class_set, prob_model, sys_output)
    write_system_output(training_result, test_result, sys_output)
    print_confusion_matrix(training_result[1], training_data[1], test_result[1], test_data[1])



