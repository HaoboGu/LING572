import sys
import os
from operator import itemgetter
import numpy as np
from math import exp
from math import tanh


class SVM:
    def __init__(self):
        self.sv = []
        self.svm_type = ''
        self.kernel = ''  # linear, polynomial, rbf, sigmoid
        self.gamma = 0
        self.coef = 0
        self.degree = 0
        self.n_total_sv = 0
        self.nr_class = 0
        self.rho = 0
        self.label = [0, 1]
        self.nr_sv = [0, 0]
        self.n_feat = 0

    def set_svm_type(self, svm_type):
        self.svm_type = svm_type

    def set_kernel_type(self, kernel_type):
        self.kernel = kernel_type

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_coef(self, coef):
        self.coef = coef

    def set_degree(self, degree):
        self.degree = degree

    def set_total_sv(self, total_sv):
        self.n_total_sv = total_sv

    def set_nr_class(self, nr_class):
        self.nr_class = nr_class

    def set_rho(self, rho):
        self.rho = rho

    def set_label(self, label):
        self.label = label

    def set_nr_sv(self, nr_sv):
        self.nr_sv = nr_sv

    def set_sv(self, sv):
        self.sv = sv

    def set_n_feat(self, n_feat):
        self.n_feat = n_feat

    def _inner_product(self, v1, v2):
        """
        Calculate inner product <v1,v2> according to the SVM's kernel type and parameters
        :param v1: feature dictionary
        :param v2: feature dictionary
        :return:
        """
        if self.kernel == 'linear':
            return np.inner(v1, v2)
        elif self.kernel == 'polynomial':
            return (self.gamma * np.inner(v1, v2) + self.coef) ** self.degree
        elif self.kernel == 'rbf':
            return exp(-self.gamma * ((np.linalg.norm(v1-v2))**2))
        elif self.kernel == 'sigmoid':
            return tanh(self.gamma*np.inner(v1, v2) + self.coef)

    def decode(self, data):
        """
        Run svm classification on dataz
        :param data:
        :return: predicted results, (truth, predicted, f(x))
        """
        re = []
        for label, feat_vec in data:
            s = 0
            for weight, support_vec in self.sv:
                s += weight * self._inner_product(support_vec, feat_vec)
            s = s - self.rho
            if s >= 0:
                re.append((label, 0, s))
            else:
                re.append((label, 1, s))

        return re


def read_svm(model_file):
    """
    Read svm model from model file.
    :param model_file: filename
    :return:
    """
    svm = SVM()
    with open(model_file, 'r') as svm_file:
        line = svm_file.readline().strip('\n')
        while line:
            # First read parameters
            if line == 'SV':
                break
            else:
                params = line.split(' ')
                if params[0] == 'svm_type':
                    svm.set_svm_type(params[1])
                elif params[0] == 'kernel_type':
                    svm.set_kernel_type(params[1])
                elif params[0] == 'gamma':
                    svm.set_gamma(float(params[1]))
                elif params[0] == 'coef0':
                    svm.set_coef(float(params[1]))
                elif params[0] == 'degree':
                    svm.set_degree(float(params[1]))
                elif params[0] == 'nr_class':
                    svm.set_nr_class(int(params[1]))
                elif params[0] == 'total_sv':
                    svm.set_total_sv(int(params[1]))
                elif params[0] == 'rho':
                    svm.set_rho(float(params[1]))
                elif params[0] == 'label':
                    svm.set_label([int(params[1]), int(params[2])])
                elif params[0] == 'nr_sv':
                    svm.set_nr_sv([int(params[1]), int(params[2])])
            line = svm_file.readline().strip('\n')

        # Then read support vectors
        line = svm_file.readline().strip('\n')
        sv = []
        max_feat_index = 0
        while line:
            seq = line.strip(' ').split(' ')
            weight = float(seq[0])  # read weight
            # Read support vector
            support_vector = {}
            for feature in seq[1:]:
                feat_index, feat_val = feature.split(':')
                support_vector[int(feat_index)] = float(feat_val)
            max_feat_index = max(max_feat_index, max(support_vector.keys()))
            sv.append((weight, support_vector))
            line = svm_file.readline().strip('\n')

        svm.set_n_feat(max_feat_index + 1)

        full_sv = []
        for weight, vector in sv:
            for i in range(0, svm.n_feat):
                if i not in vector:
                    vector[i] = 0
            sorted_sv = np.array(sorted(vector.items(), key=itemgetter(0)))
            # print(sorted_sv.transpose())
            full_sv.append((weight, sorted_sv.transpose()[1]))
        svm.set_sv(full_sv)
    return svm


def read_test_data(filename, n_features):
    """
    Read test data from test data file
    :param filename:
    :param n_features: number of features in training data
    :return:
    """
    data = []
    with open(filename, 'r') as test_file:
        line = test_file.readline().strip('\n')
        while line:
            seq = line.strip(' ').split(' ')
            gold = int(seq[0])  # read label
            # Read feature vector
            feature_vector = {}
            for feature in seq[1:]:
                feat_index, feat_val = feature.split(':')
                feature_vector[int(feat_index)] = float(feat_val)
            for i in range(0, n_features):
                if i not in feature_vector:
                    feature_vector[i] = 0
            sorted_sv = np.array(sorted(feature_vector.items(), key=itemgetter(0)))[:n_features]
            data.append((gold, sorted_sv.transpose()[1]))
            line = test_file.readline().strip('\n')

    return data


def write_result(re, output_file):
    with open(output_file, 'w') as of:
        i = 0
        for truth, predicted, val in re:
            out_str = 'array:' + str(i) + ' ' + str(truth) + ' ' + str(predicted) + ' ' + '%0.5f' % val + '\n'
            of.write(out_str)
            i += 1


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        if 'hw8' in os.listdir():
            os.chdir('hw8')
        test_data_filename = 'examples/test'
        model_filename = 'model.5'
        sys_output = 'sys.5'
    else:
        test_data_filename = sys.argv[1]
        model_filename = sys.argv[2]
        sys_output = sys.argv[3]

    print('Reading SVM...')
    svm_model = read_svm(model_filename)
    print('Reading test data...')
    test_data = read_test_data(test_data_filename, svm_model.n_feat)
    print('Decoding...')
    result = svm_model.decode(test_data)
    write_result(result, sys_output)

