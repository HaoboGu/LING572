import sys
import os
from math import exp
from operator import itemgetter
from math import log10
import time


class Inst:
    def __init__(self, sent_num, instance_name, pos, feature_set, label='', cur_prob=0, path_prob=0.0, prev_node=None):
        self.sent = sent_num
        self.name = instance_name
        self.pos = pos
        self.features = feature_set
        self.predicted_label = label
        self.cur_prob = cur_prob
        self.path_prob = path_prob
        self.prev_node = prev_node


def generate_added_features(word_index, pre_node):
    """
    Generate prevT and prevTwoTags features for current word
    :return:
    """
    if word_index == 0:
        return {'prevT=BOS', 'prevTwoTags=BOS+BOS'}
    elif word_index == 1:
        tag1 = 'prevT=' + pre_node.predicted_label
        tag2 = 'prevTwoTags=BOS+' + pre_node.predicted_label
        return {tag1, tag2}
    else:
        tag1 = 'prevT=' + pre_node.predicted_label
        tag2 = 'prevTwoTags=' + pre_node.prev_node.predicted_label + '+' + pre_node.predicted_label
        return {tag1, tag2}


def prune(result, top_k, beam_size):
    """
    Prune useless nodes in result list.
    :param result: [(label, cur_prob, path_prob, prev_node)]
    :param top_k: keep top k nodes
    :param beam_size: keep nodes when lg(p) + beam_size >= lg(max_p)
    :return:
    """
    sorted_result = sorted(result, key=itemgetter(2), reverse=True)[:top_k]
    max_prob = sorted_result[0][2]
    final_result = []
    for label, cur_prob, path_prob, prev_node in sorted_result:
        if path_prob + beam_size >= max_prob:
            final_result.append((label, cur_prob, path_prob, prev_node))
    return final_result


def beam_maxent_decoder(data, model, bound_list, labels, top_n, top_k, beam_size, output_file):
    """
    Maxent decoder for POS tagging implemented using beam search.
    :param data: test data
    :type data: list[Inst]
    :param model: maxent model
    :param bound_list: bounds for every sentence
    :param labels: label set
    :param top_n: keep top n tags for each node
    :param top_k: keep top k nodes at each position
    :param beam_size: max gap between lg_prob of the best path and kept path
    :param output_file: output file
    :return:
    """
    correct, i, sent_index, word_index = 0, 0, 0, 0
    prev_nodes = []
    out_f = open(output_file, 'w')
    out_f.write("%%%%% test data:\n")
    while i < len(data):
        if word_index == 0:  # first word in the sentence
            cur_features = data[i].features
            cur_features.union(generate_added_features(word_index, None))  # None previous node for first word
            result = calculate_p(model, labels, cur_features)[:top_n]  # keep top n results
            prev_nodes = []
            for label, prob in result:
                # Create a new set of nodes
                new_node = Inst(data[i].sent, data[i].name, data[i].pos, data[i].features, label, prob, log10(prob), None)
                prev_nodes.append(new_node)
        else:
            results = []
            for prev_node in prev_nodes:
                # For every previous node, form features first
                cur_features = data[i].features
                cur_features = cur_features.union(generate_added_features(word_index, prev_node))
                # Calculate p(y|x), choose top n of them
                cur_p = calculate_p(model, labels, cur_features)[:top_n]  # top_n predicted tags and probs
                cur_result = []  # list of (label, cur_prob, path_prob, prev_node)
                for l, p in cur_p:
                    # Calculate path's probability and add previous tag to the result
                    cur_result.append((l, p, log10(p)+prev_node.path_prob, prev_node))
                results.extend(cur_result)

            final_results = prune(results, top_k, beam_size)
            prev_nodes = []
            for label, cur_prob, path_prob, prev_node in final_results:
                # Create a new set of nodes
                new_node = Inst(data[i].sent, data[i].name, data[i].pos, data[i].features,
                                label, cur_prob, path_prob, prev_node)
                prev_nodes.append(new_node)

        if word_index >= bound_list[sent_index]-1:
            # last word in the sentence
            cur_node = prev_nodes[0]
            out_strings = []
            while cur_node:
                out_s = cur_node.name + ' ' + cur_node.pos + ' ' + cur_node.predicted_label + ' ' + \
                        "%.5f" % cur_node.cur_prob + '\n'
                out_strings.append(out_s)
                if cur_node.predicted_label == cur_node.pos:
                    correct += 1
                cur_node = cur_node.prev_node
            out_strings.reverse()
            for item in out_strings:
                out_f.write(item)
        if word_index >= bound_list[sent_index]-1:
            sent_index += 1
            word_index = 0
        else:
            word_index += 1
        i += 1
    out_f.close()
    print(correct/len(data))


def calculate_p(model, labels, features):
    """
    Calculate p(y|x) for instance x and each label y.
    :param model: 
    :param labels:
    :param features: features of instance x
    :return: list[(label, probability)]
    """
    results = {}
    for label in labels:
        sum_exp = model[(label, '<default>')]
        for feat in features:
            if (label, feat) in model:
                sum_exp += model[(label, feat)]
        results[label] = exp(sum_exp)

    z = sum(results.values())
    for label in labels:
        results[label] = results[label] / z
    sorted_results = sorted(results.items(), key=itemgetter(1), reverse=True)  # sort results by probability
    return sorted_results


def read_boundary(filename):
    """
    Read boundary, which is the length of each sentence
    :param filename:
    :return: list[int]
    """
    boundary_list = []
    with open(filename, 'r') as b_file:
        line = b_file.readline().strip('\n')
        while line:
            boundary_list.append(int(line))
            line = b_file.readline().strip('\n')
    return boundary_list


def read_data(filename, bound_list):
    """
    Read data
    :param filename: str
    :param bound_list: list[int]
    :return: np.array([label, feature]), set of features, set of labels
    """
    data = []
    sent_index = 0
    word_index = 0
    with open(filename, 'r') as read_file:
        line = read_file.readline().strip('\n')
        while line:
            if word_index >= bound_list[sent_index]:
                sent_index += 1
                word_index = 0
            else:
                word_index += 1
            seq = line.split(' ')
            features = set(seq[2::2])
            data.append(Inst(sent_index, seq[0], seq[1], features))
            line = read_file.readline().strip('\n')
    return data


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


if __name__ == "__main__":
    s = time.time()
    use_local_file = False
    if use_local_file:
        if 'hw6' in os.listdir():
            os.chdir('hw6')
        test_data_filename = 'examples/sec19_21.txt'
        boundary_filename = 'examples/sec19_21.boundary'
        model_filename = 'examples/m1.txt'
        sys_output = 'hw6_output.txt'
        bs = 1
        n = 3
        k = 5
    else:
        test_data_filename = sys.argv[1]
        boundary_filename = sys.argv[2]
        model_filename = sys.argv[3]
        sys_output = sys.argv[4]
        bs = int(sys.argv[5])
        n = int(sys.argv[6])
        k = int(sys.argv[7])

    # key of the mode: (POS, feature)
    model_dict, label_set = read_model(model_filename)

    bounds = read_boundary(boundary_filename)

    test_data = read_data(test_data_filename, bounds)

    beam_maxent_decoder(test_data, model_dict, bounds, label_set, n, k, bs, sys_output)

    e = time.time()
    print('Running time:', (e-s)/60, 'min')