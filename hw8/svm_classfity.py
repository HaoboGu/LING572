import sys
import os


class SVM:
    def __init__(self):
        self.sv = []
        self.svm_type = ''
        self.kernel = ''
        self.gamma = 0
        self.coef = 0
        self.degree = 0
        self.n_total_sv = 0
        self.nr_class = 0
        self.rho = 0
        self.label = [0, 1]
        self.nr_sv = [0, 0]

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

    def predict(self, data):
        """
        Run svm classification on data
        :param data:
        :return: predicted labels
        """
        result = []
        return result


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
    return svm


__name__ = "__main__"
if __name__ == "__main__":
    use_local_file = True
    if use_local_file:

        if 'hw8' in os.listdir():
            os.chdir('hw8')
        test_data_filename = 'examples/test'
        model_filename = 'examples/model_ex'
        sys_output = 'sys.out'
    else:
        test_data_filename = sys.argv[1]
        model_filename = sys.argv[2]
        sys_output = sys.argv[3]

    svm_model = read_svm(model_filename)




