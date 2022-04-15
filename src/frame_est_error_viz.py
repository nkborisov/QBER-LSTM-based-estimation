import csv
import matplotlib.pyplot as plt
import numpy as np
from sandbox.model.error_est import qber_estimation
import math
from ema import *

def get_kalman_gain(sigma_ema_val, sigma_est_val):
    if not sigma_ema_val or not sigma_est_val:
        return 0.1
    return np.sqrt(sigma_ema_val * sigma_ema_val / (sigma_est_val * sigma_est_val + sigma_ema_val * sigma_ema_val))


def get_kalman_gain2(sigma_ema_val, e):
    if not sigma_ema_val or not e:
        return 0.35
    return e * e / (sigma_ema_val * sigma_ema_val)


def get_error_opt(sigma_ema_val, sigma_est_val, e_prev):
    if not sigma_ema_val or not sigma_est_val:
        return 0.
    return np.sqrt(sigma_ema_val * sigma_ema_val * (e_prev * e_prev + sigma_est_val * sigma_est_val) /
                   (sigma_est_val * sigma_est_val + e_prev * e_prev + sigma_ema_val * sigma_ema_val))


e_mu_est_eval = list()
e_mu_est_by_code = list()
e_mu_act = list()
e_mu_weight_est = list()
ema_est_list = list()
var_window_size = 500
sigma_est = VarianceEstimator2(var_window_size)
sigma_ema = VarianceEstimator2(var_window_size)
delta_weight = list()
delta_est = list()
delta_ema = list()
rounds_count = list()
gains_list = list()
delta = list()
q_mu = list()
q_nu1 = list()
q_nu2 = list()
e_nu1 = list()
e_nu2 = list()
e_prev = None

path = '/home/galactic/profiling/26-11-2021/frames_errors.csv'
ema_avg = None
with open(path, 'r') as csv_file:
    rows = csv.reader(csv_file, delimiter=',')
    i = 0
    for row in rows:
        e_nu1_val = float(row[4])
        if ema_avg is None:
            ema_avg = ExponentionalAverager(e_nu1_val, 5)
        e_nu1.append(e_nu1_val)
        e_nu2_val = float(row[5])
        e_nu2.append(e_nu2_val)
        e_mu_act_val = float(row[2])
        ema_val = ema_avg.get_value()
        ema_est_list.append(ema_val)
        ema_avg.add_value(e_mu_act_val)
        # frame stats
        m_mu = float(row[9])
        n_mu = float(row[6])
        m_nu1 = float(row[10])
        n_nu1 = float(row[7])
        m_nu2 = float(row[11])
        n_nu2 = float(row[8])
        e_mu_est_val = qber_estimation.q_e_4(n_mu, m_mu, e_mu_act_val, n_nu1, m_nu1, e_nu1_val, n_nu2, m_nu2, e_nu2_val, z=-0.4)
        e_mu_act.append(e_mu_act_val)
        e_mu_est_eval.append(e_mu_est_val)
        e_mu_est_by_code_val = float(row[3])
        e_mu_est_by_code.append(e_mu_est_by_code_val)
        delta_est_val = e_mu_act_val - e_mu_est_val
        delta_ema_est = e_mu_act_val - ema_val
        sigma_ema_val = sigma_ema.retrieve_sigma()
        sigma_est_val = sigma_est.retrieve_sigma()
        if e_prev is None:
            e_prev = sigma_ema_val
        e = get_error_opt(sigma_est_val, sigma_ema_val, e_prev)
        k = get_kalman_gain2(sigma_est_val, e)
        e_prev = e
        gains_list.append(k)
        q_mu_val = n_mu / m_mu * 10 ** 3
        q_nu1_val = n_nu1 / m_nu1 * 10 ** 3
        q_nu2_val = n_nu2 / m_nu2 * 10 ** 3
        q_mu.append(q_mu_val)
        q_nu1.append(q_nu1_val)
        q_nu2.append(q_nu2_val)
        e_mu_weight_est_val = k * e_mu_est_val + (1. - k) * ema_val
        delta_weight_val = e_mu_act_val - e_mu_weight_est_val
        delta_ema_val = e_mu_act_val - ema_val
        delta_weight_percent = abs(delta_weight_val) * 100. / e_mu_act_val
        delta_est_percent = abs(delta_est_val) * 100. / e_mu_act_val
        delta_ema_percent = abs(delta_ema_val) * 100. / e_mu_act_val
        delta_weight.append(delta_weight_percent)
        delta_est.append(delta_est_percent)
        delta_ema.append(delta_ema_percent)
        delta.append(delta_weight_percent - delta_est_percent)
        e_mu_weight_est.append(e_mu_weight_est_val)
        sigma_ema.add_value(delta_ema_est)
        sigma_est.add_value(delta_est_val)
        rounds_count.append(math.log10(float(row[12])) / 10)
        i += 1

avg_gain = np.mean(gains_list)
print("Average Kalman gain value: {}".format(avg_gain))
print("Average weight delta value: {}%".format(np.mean(delta_weight)))
print("Average EMA delta value: {}%".format(np.mean(delta_ema)))

n_beg = 0
n_end = len(e_mu_est_eval)

block_indices = [i for i in range(len(e_mu_est_eval))]
fig, axs = plt.subplots(4)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 10})

axs[0].set_title('Kalman-weighted estimation vs actual value')
axs[0].plot(block_indices[n_beg:n_end], e_mu_act[n_beg:n_end], 'b', label=r'$E_\mu$')
axs[0].plot(block_indices[n_beg:n_end], e_mu_weight_est[n_beg:n_end], 'g', label=r'$E_\mu$ (weight est)')

axs[0].grid()
axs[1].plot(block_indices[n_beg:n_end], ema_est_list[n_beg:n_end], 'purple', label=r'$E_\mu$ (EMA est)')
axs[1].plot(block_indices[n_beg:n_end], e_mu_act[n_beg:n_end], 'b', label=r'$E_\mu$')
axs[1].grid()

axs[2].plot(block_indices[n_beg:n_end], gains_list[n_beg:n_end], 'aqua', label=r'Kalman gain')
axs[2].grid()

delta_e = [e_nu1[i] - e_mu_act[i] for i in range(0, len(e_mu_act))]
axs[3].plot(block_indices[n_beg:n_end], delta_weight[n_beg:n_end], 'fuchsia', label=r'$\delta_{weight}$')
axs[3].plot(block_indices[n_beg:n_end], delta_ema[n_beg:n_end], 'purple', label=r'$\delta_{EMA}$')

axs[3].grid()

fig.legend(fontsize=10, loc='upper right')
plt.show()
