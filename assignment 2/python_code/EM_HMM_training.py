import numpy as np
from util import HMM
import csv
import pickle as pkl
import matplotlib.pyplot as plt
from EM_HMM import EM_HMM


def EM_HMM_training(training_file='./csvfiles/training_csv', DIM=39, num_model=10, num_state=13):
    """
    :param training_file:
    :param DIM:
    :param num_model:
    :param num_state:
    :return:
    """
    hmm = HMM()
    hmm.mean = np.full((num_model, DIM, num_state), 0.)
    hmm.var = np.full((num_model, DIM, num_state), 0.)
    hmm.aij = np.full((num_model, num_state + 2, num_state + 2), 0.)
    hmm = initial_EM(hmm, DIM, num_state, num_model, training_file)

    num_iter = 20  # bigger than 10
    log_likelihood_iter = np.full(num_iter, 0.)
    likelihood_iter = np.full(num_iter, 0.)
    x = [i for i in range(1, num_iter + 1)]
    for itr in range(num_iter):
        with open(training_file) as f:
            # initialization
            sum_mean_numerator = np.full((num_model, DIM, num_state), 0.)
            sum_var_numerator = np.full((num_model, DIM, num_state), 0.)
            sum_aij_numerator = np.full((num_model, num_state, num_state,), 0.)
            sum_denominator = np.full((num_state, num_model), 0.)
            log_likelihood = 0
            likelihood = 0

            # read the csv file
            reader = csv.reader(f)
            for row in reader:
                label = int(row[0])
                file = row[1]
                with open(file, 'rb') as data:
                    dic = pkl.load(data)
                    for feature in dic.values():
                        mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood_i, likelihood_i = \
                            EM_HMM(hmm.mean[label - 1, :, :], hmm.var[label - 1, :, :], hmm.aij[label - 1, :, :],
                                   feature)

                        sum_mean_numerator[label - 1, :, :] += mean_numerator[:, 1:-1]
                        sum_var_numerator[label - 1, :, :] += var_numerator[:, 1:-1]
                        sum_aij_numerator[label - 1, :, :] += aij_numerator[1:-1, 1:-1]
                        sum_denominator[:, label - 1] += denominator[1:-1].flatten()
                        log_likelihood += log_likelihood_i
                        likelihood += likelihood_i
            for k in range(num_model):
                for n in range(num_state):
                    hmm.mean[k, :, n] = sum_mean_numerator[k, :, n] / sum_denominator[n, k]
                    hmm.var[k, :, n] = sum_var_numerator[k, :, n] / sum_denominator[n, k] - hmm.mean[k, :,
                                                                                            n] * hmm.mean[k,
                                                                                                 :, n]
            for k in range(num_model):
                for i in range(1, num_state + 1):
                    for j in range(1, num_state + 1):
                        hmm.aij[k, i, j] = sum_aij_numerator[k, i - 1, j - 1] / sum_denominator[i - 1, k]
                hmm.aij[k, num_state, num_state + 1] = 1 - hmm.aij[k, num_state, num_state]
            hmm.aij[k, num_state + 1, num_state + 1] = 1
            log_likelihood_iter[itr] = log_likelihood
            likelihood_iter[itr] = likelihood

    # draw a figure
    plt.figure()
    plt.plot(x, log_likelihood_iter)
    plt.xlabel("iterations")
    plt.ylabel("log likelihood")
    plt.title("number of states: " + str(num_state))
    plt.show()

    # save the model as pickle file
    with open('./HMM_model/hmm.pkl', 'wb') as f:
        pkl.dump(hmm, f)
    return hmm


def initial_HMM(hmm: HMM, num_of_state, num_of_model, sum_of_features, sum_of_features_square,
                num_of_feature):
    """
    :param hmm:
    :param num_of_state:
    :param num_of_model:
    :param sum_of_features:
    :param sum_of_features_square:
    :param num_of_feature:
    :return:
    """
    for k in range(0, num_of_model):
        for m in range(0, num_of_state):
            hmm.mean[k, :, m] = sum_of_features.T / num_of_feature
            hmm.var[k, :, m] = sum_of_features_square.T / num_of_feature - hmm.mean[k, :, m] * hmm.mean[k, :, m]
        for i in range(1, num_of_state + 1):
            hmm.aij[k, i, i + 1] = 0.4
            hmm.aij[k, i, i] = 1 - hmm.aij[k, i, i + 1]
        hmm.aij[k, 0, 1] = 1
    return hmm


def initial_EM(hmm: HMM, DIM, num_of_state, num_of_model, training_file='./csvfiles/training.csv'):
    """
    :param hmm:
    :param DIM:
    :param num_of_state:
    :param num_of_model:
    :param training_file:
    :return:
    """
    sum_of_feature = np.full((DIM, 1), 0.)
    sum_of_feature_square = np.full((DIM, 1), 0.)
    num_of_feature = 0
    with open(training_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                pass
            file = row[1]
            with open(file, 'rb') as data:
                dic = pkl.load(data)
                for feature in dic.values():
                    sum_of_feature += np.sum(feature, 1, keepdims=True)
                    sum_of_feature_square += np.sum(feature ** 2, 1, keepdims=True)
                    num_of_feature += feature.shape[1]

        # initialize value of means, variances, aijs
    hmm = initial_HMM(hmm, num_of_state, num_of_model, sum_of_feature, sum_of_feature_square, num_of_feature)
    return hmm
