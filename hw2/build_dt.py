import sys

def read_data(filename):
    f = open(filename)
    line = f.readline().strip('\n')
    data = []
    while line:
        tokens = line.split(' ')
        label = tokens[0]
        features = tokens[1:]
        data.append([features, label])
        line = f.readline().strip('\n')
    print(data)
    return data



if __name__ == "__main__":
    use_local_file = True
    if use_local_file:
        training_data_filename = 'examples/train.vectors.txt'
        test_data_filename = 'examples/test_data'
        max_depth = 2
        min_gain = 0
        model_file = 'examples/model_file'
        sys_output = 'sys.out'

    elif len(sys.argv) == 7:
        training_data = sys.argv[1]
        test_data = sys.argv[2]
        max_depth = sys.argv[3]
        min_gain = sys.argv[4]
        model_file = sys.argv[5]
        sys_output = sys.argv[6]

    else:
        print('The number of argvs is not correct')

    training_data = read_data(training_data_filename)
    test_data = read_data(test_data_filename)
