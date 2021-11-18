from util import *


def EM_HMM(mean, var, aij, obs):
    """
    the EM-HMM algorithm
    :param mean:
    :param var:
    :param aij:
    :param obs:
    :return:
    """
    # initialization
    dim, T = obs.shape
    nan_array = np.full((dim, 1), np.nan)
    mean = np.concatenate((nan_array, mean), 1)
    mean = np.concatenate((mean, nan_array), 1)

    var = np.concatenate((nan_array, var), 1)
    var = np.concatenate((var, nan_array), 1)

    N = mean.shape[1]

    # Step 1: calculate alpha
    log_alpha = np.full((N, T + 1), -np.inf)
    for i in range(0, N):
        log_alpha[i, 0] = my_log(aij[0, i]) + log_Gaussian(mean[:, i], var[:, i], obs[:, i])  # log(alpha)
    for t in range(1, T):
        for j in range(1, N - 1):  # inner not corresponded with outer
            log_alpha[j, t] = log_sum_alpha(log_alpha[1:N - 1, t - 1], aij[1:N - 1, j]) + \
                              log_Gaussian(mean[:, j], var[:, j], obs[:, t])
    log_alpha[N - 1, T] = log_sum_alpha(log_alpha[1:N - 1, T - 1], aij[1:N - 1, N - 1])  # this value is  P(o1, o2,...
    # , oT | lambda) also

    # Step 2: calculate beta
    log_beta = np.full((N, T + 1), -np.inf)
    log_beta[:, T - 1] = my_log(aij[:, N - 1])
    for t in range(T - 2, -1, -1):
        for i in range(1, N - 1):
            log_beta[i, t] = log_sum_beta(aij[0, 1:N - 1], mean[:, 1:N - 1], var[:, 1:N - 1], obs[:, t + 1],
                                          log_beta[1:N - 1, t + 1])
    log_beta[N - 1, 0] = log_sum_beta(aij[1, 2:N - 1], mean[:, 2:N - 1], var[:, 2:N - 1], obs[:, 1],
                                      log_beta[2:N - 1, 1])

    # Step 3: calculate Xi
    # t < T-1
    log_Xi = log_beta = np.full((T, N, N), -np.inf)
    for t in range(0, T - 1):
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                log_Xi[t, i, j] = log_alpha[i, t] + my_log(aij(i, j)) + \
                                  log_Gaussian(mean[:, j], var[:, j], obs[:, t + 1]) + \
                                  log_beta[j, t + 1] - \
                                  log_alpha[N - 1, T]

    # t == T-1
    for i in range(0, N):
        log_Xi[T, i, N] = log_alpha[i, T] + my_log(aij[i, N]) - log_alpha[N - 1, T]

    # Step 4: calculate gamma
    log_gamma = np.full((N, T), -np.inf)
    for t in range(0, T):
        for i in range(1, N - 1):
            log_gamma[i, t] = log_alpha[i, t] + log_beta[i, t] - log_alpha[N - 1, T]
    gamma = np.exp(log_gamma)

    # Step 5: calculate numerators, denominator and likelihood

    # initialization
    mean_numerator = np.full((dim, N), 0)
    var_numerator = np.full((dim, N), 0)
    aij_numerator = np.full((N, N), 0)
    denominator = np.full((N, 1), 0)

    # mean&var numerator and denominator
    for j in range(1, N - 1):
        for t in range(0, T):
            mean_numerator[:, j] += np.dot(gamma[j, t], obs[:, t])
            var_numerator[:, j] += np.dot(gamma[j, t], obs[:, t]) * obs[:, t]
            denominator[j] += gamma[j, t]
    # aij numerator
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for t in range(0, T):
                aij_numerator[i, j] += np.exp(log_Xi[t, i, j])

    log_likelihood = log_alpha[N - 1, T]
    likelihood = np.exp(log_likelihood)

    return mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood, likelihood
