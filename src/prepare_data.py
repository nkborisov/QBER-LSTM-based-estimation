import csv
from ema import ExponentionalAverager


source_path = "/home/galactic/profiling/stats/fr.csv"
target_path = "/home/galactic/profiling/stats/fr_gains.csv"

e_mu = list()
e_nu1 = list()
e_nu2 = list()
q_mu = list()
q_nu1 = list()
q_nu2 = list()
e_mu_est = list()
ema_est = list()
avg = ExponentionalAverager(0.01, 5)

with open(source_path, 'r') as source_file:
    table = csv.reader(source_file, delimiter=',')
    n_frame_err = 0
    for row in table:
        e_mu_cur = float(row[2])
        e_mu.append(e_mu_cur)
        ema_est.append(avg.get_value())
        avg.add_value(e_mu_cur)
        e_mu_est.append(float(row[3]))
        e_nu1.append(float(row[4]))
        e_nu2.append(float(row[5]))
        q_mu_val = float(row[9]) / float(row[6]) * 1000.
        q_nu1_val = float(row[10]) / float(row[7]) * 1000.
        q_nu2_val = float(row[11]) / float(row[8]) * 1000.
        q_mu.append(q_mu_val)
        q_nu1.append(q_nu1_val)
        q_nu2.append(q_nu2_val)

rows = list(zip(e_mu, e_nu1, e_nu2, e_mu_est, q_mu, q_nu1, q_nu2))

with open(target_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
