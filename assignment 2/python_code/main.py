import pickle as pkl

import numpy as np
from util import HMM
from EM_HMM_training import EM_HMM_training
from HMM_testing import HMM_testing

if __name__ == '__main__':
    training_file = './csvfiles/training.csv'
    testing_file = './csvfiles/testing.csv'
    DIM = 39
    num_of_model = 11
    num_of_state_start = 12
    num_of_state_end = 15
    accuracy_rate = np.zeros(num_of_state_end)
    for state in range(num_of_state_start, num_of_state_end + 1):
        hmm = EM_HMM_training(training_file, DIM, num_of_model, state)
        accuracy_rate[state - 1] = HMM_testing(hmm, testing_file)
        print("state:%d, accuracy rate:%f" % (state, accuracy_rate[state - 1]))
    # with open("./HMM_model/hmm.pkl", 'rb') as f:
    #     hmm = pkl.load(f)
    # accuracy = HMM_testing(hmm)
    # print(accuracy)
