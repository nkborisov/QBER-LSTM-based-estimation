import numpy as np

"""
Adaptive estimation with upper bound
"""


def q_e_4(N_mu, M_mu, e_mu,
          N_nu1, M_nu1, e_nu1,
          N_nu2, M_nu2, e_nu2, z=1):
    sigma = lambda x, y: np.sqrt(x * (y - x) / y)

    M_mu_u = M_mu + z * sigma(M_mu, N_mu)
    M_mu_l = M_mu - z * sigma(M_mu, N_mu)

    M_nu1_u = M_nu1 + z * sigma(M_nu1, N_nu1)
    M_nu1_l = M_nu1 - z * sigma(M_nu1, N_nu2)
    M_nu1_err = M_nu1 * e_nu1
    M_nu1_err_u = M_nu1_err + z * sigma(M_nu1_err, N_nu1)

    M_nu2_u = M_nu2 + z * sigma(M_nu2, N_nu2)
    M_nu2_l = M_nu2 - z * sigma(M_nu2, N_nu2)
    M_nu2_err = M_nu2 * e_nu2
    M_nu2_err_l = M_nu2_err - z * sigma(M_nu1_err, N_nu1)

    nmrtr = M_nu1_err_u / N_nu1 * (1 - M_nu2_l / M_mu_u * N_mu / N_nu2) - \
            M_nu2_err_l / N_nu2 * (1 - M_nu1_u / M_mu_l * N_mu / N_nu2)

    dnmrtr = M_nu1_l / N_nu1 - M_nu2_u / N_nu2

    return nmrtr / dnmrtr
