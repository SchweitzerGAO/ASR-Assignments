import numpy as np
from util import HMM
from viterbi import viterbi
import csv
import pickle as pkl


def HMM_testing(hmm: HMM, testing_file='./csvfiles/testing.csv'):
    num_model = 11
    num_error = 0
    num_test = 0
    with open(testing_file) as f:
        reader = csv.reader(f)
        for row in reader:
            num_test += 1
            label = row[0]
            label = int(label)
            file = row[1]
            with open(file, 'rb') as data:
                dic = pkl.load(data)
                for feature in dic.values():
                    # predicting HAHA!
                    fopt_max = -np.inf
                    digit = -1
                    for p in range(1, num_model + 1):
                        fopt = viterbi(hmm.mean[p - 1, :, :], hmm.var[p - 1, :, :], hmm.aij[p - 1, :, :], feature)
                        if fopt > fopt_max:
                            digit = p
                            fopt_max = fopt
                    if digit != label:
                        num_error += 1
    return (num_test-num_error)*100/num_test
