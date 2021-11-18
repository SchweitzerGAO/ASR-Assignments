import numpy as np


class HMM:
    def __init__(self):
        self.mean = None
        self.var = None
        self.aij = None


# Gaussian-log function
def log_Gaussian(mean_i, var_i, o_i):
    dim = np.max(var_i.shape)
    return (-1 / 2) * (
            dim * np.log(2 * np.pi) + np.sum(np.log(var_i)) + np.sum((o_i - mean_i) * (o_i - mean_i) / var_i))


def my_log(x):
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


def parse(array, value):
    temp = []
    for item in array:
        temp.append(item)
    temp.append(value)
    return temp


def log_sum_alpha(log_alpha_t, aij_j):
    len_x = log_alpha_t.shape[0]
    y = np.full((1, len_x), -np.inf)
    y_max = -np.inf
    for i in range(0, len_x):
        y[i] = log_alpha_t[i] + my_log(aij_j[i])
        if y[i] > y_max:
            y_max = y[i]
    return sum_exp(y, y_max, len_x)


def log_sum_beta(aij_i, mean, var, obs, log_beta_t):
    len_x = mean.shape[1]
    y = np.full((1, len_x), -np.inf)
    y_max = -np.inf
    for j in range(0, len_x):
        y[j] = my_log(aij_i[j]) + log_Gaussian(mean[:, j], var[:, j], obs) + log_beta_t[j]
        if y[j] > y_max:
            y_max = y[j]
    return sum_exp(y, y_max, len_x)


def sum_exp(y, y_max, len_x):
    if y_max == -np.inf:
        return y_max
    else:
        ret = 0
        for i in range(0, len_x):
            if y_max == -np.inf and y[i] == -np.inf:
                ret += 1
            else:
                ret += np.exp(y[i] - y_max)
        return y_max + my_log(ret)

