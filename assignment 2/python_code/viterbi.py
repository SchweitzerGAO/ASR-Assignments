# 译事三难：信，达，雅——严复

import numpy as np

# Viterbi Alg in ASR translated from MATLAB code by Min-Lee Lee and Hoang-Hiep Le
# repo: https://github.com/hhle88/HMM


from util import log_Gaussian, my_log,parse





# default parameter values for Viterbi Alg

# mean
default_mean = np.array([[10., 0., 0.],
                         [5., 2., 9.]])
# variance
default_var = np.array([[1., 1., 1.],
                        [1., 1., 1.]])

# observation
default_obs = np.array([[8., 8., 4., 2., 3., 7.],
                        [0., 2., 2., 10., 5., 9.]])

A = np.array([[0., 1., 0., 0., 0.],
              [0., 0.5, 0.5, 0., 0.],
              [0., 0., 0.5, 0.5, 0.],
              [0., 0., 0., 0.5, 0.5],
              [0., 0., 0., 0., 1.]])

# emission matrix
# concatenate 2-D array to 3-D
default_aij = np.array([A, np.dot(A, A), np.dot(np.dot(A, A), A)])  # for loss frame case


def viterbi(mean=default_mean, var=default_var, aij=default_aij, obs=default_obs):
    """
    the viterbi algorithm in Python
    :param mean: 均值
    :param var: 方差
    :param aij: 发射矩阵的[i,j]元素
    :param obs: 观察矩阵
    :return: 最优预测结果
    """
    dim, t_len = obs.shape
    nan_array = np.full((dim, 1), np.nan)

    # initialize the table
    mean = np.concatenate((nan_array, mean), 1)
    mean = np.concatenate((mean, nan_array), 1)

    var = np.concatenate((nan_array, var), 1)
    var = np.concatenate((var, nan_array), 1)

    shape_aij = aij.shape
    aij[shape_aij[0] - 1][shape_aij[1] - 1] = 1

    timing = np.array([i for i in range(1, t_len + 2)])
    m_len = mean.shape[1]
    fjt = np.full((m_len, t_len), -np.inf)

    # the 'cell' in MATLAB is the code below
    s_chain = np.empty((m_len, t_len), dtype=object)
    # for i in range(m_len):
    #     for j in range(t_len):
    #         s_chain[i, j] = np.zeros((1, 2), dtype=object)

    # at t = 0
    dt = timing[0] - 1
    for j in range(1, m_len - 1):
        # if aij[dt, 0, j] == 0:
        #     fjt[j, 0] = -np.inf
        # else:
        #     fjt[j, 0] = np.log(aij[dt, 0, j])
        fjt[j, 0] = my_log(aij[0, j]) + log_Gaussian(mean[:, j], var[:, j], obs[:, 0])
        if fjt[j, 0] > -np.inf:
            s_chain[j, 0] = np.array([1, j + 1])

    # at t in range(1,t_len)
    for t in range(1, t_len):
        dt = timing[t] - timing[t - 1] - 1
        for j in range(1, m_len - 1):
            f_max = -np.inf
            i_max = -1
            f = -np.inf
            # index in nested loop: that is so tricky!!
            # loop in MATLAB:
            '''
                 for i=2:j  -> [2,j] note that j is equivalent to j-1 in python in terms of index
                 and the nested loop is corresponded with the outer loop
            '''
            for i in range(1, j + 1):
                if fjt[i, t - 1] > -np.inf:
                    f = fjt[i, t - 1] + my_log(aij[i, j]) + log_Gaussian(mean[:, j], var[:, j], obs[:, t])
                if f > f_max:
                    f_max = f
                    i_max = i
            if i_max != -1:
                # s_chain[j, t] = np.array([s_chain[i_max, t - 1], j])
                s_chain[j, t] = np.array(parse(s_chain[i_max, t - 1], j + 1))
                fjt[j, t] = f_max

    # at t = timing.len-1
    dt = timing[timing.shape[0] - 1] - timing[timing.shape[0] - 2] - 1

    f_opt = -np.inf
    i_opt = -1
    for i in range(1, m_len - 1):
        # if aij[dt, 0, m_len - 1] == 0:
        #     f = -np.inf
        # else:
        #     f = np.log(aij[dt, 0, m_len - 1])
        f = my_log(aij[i, m_len - 1]) + fjt[i, t_len - 1]
        if f > f_opt:
            f_opt = f
            i_opt = i

    # the optimal result
    if i_opt != -1:
        # temp = []
        # for item in s_chain[i_opt, t_len-1]:
        #     temp.append(item)
        # temp.append(m_len-1)
        chain_opt = np.array(parse(s_chain[i_opt, t_len - 1], m_len))
    return f_opt


if __name__ == '__main__':
    print(viterbi())
