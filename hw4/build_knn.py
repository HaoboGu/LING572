import sys
import numpy as np
from math import log2
from collections import Counter


if __name__ == "__main__":
    use_local_file = False
    if use_local_file:
        training_data_filename = 'examples/train.vectors.txt'
        test_data_filename = 'examples/test.vectors.txt'
        k_val = 1
        similarity_func = 0.1
        sys_output = 'sys.out'
    else:
        training_data_filename = sys.argv[1]
        test_data_filename = sys.argv[2]
        k_val = int(sys.argv[3])
        similarity_func = sys.argv[4]
        sys_output = sys.argv[5]
