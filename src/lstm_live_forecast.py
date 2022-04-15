from lstm_build_model import load_lstm_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pandas import read_csv
from ema import ExponentionalAverager
import time


scaler = MinMaxScaler(feature_range=(0, 1))


def predict_seq(test_row, model):
    test_row = np.array(test_row)
    test_row = scaler.transform(test_row)
    test_row = np.reshape(test_row, (1, 4, -1))
    predict_res = model.predict(test_row)
    predict_res = scaler.inverse_transform(predict_res).tolist()
    return predict_res[0]


def shift(seq, n):
    n = n % len(seq)
    return seq[n:] + seq[:n]


def main():
    pulses_stats_file_path = "../dataset/fr_gains.csv"
    df_whole = read_csv(pulses_stats_file_path, usecols=[0, 1, 2, 3, 4, 5, 6], engine='python')
    df = read_csv(pulses_stats_file_path, usecols=[0, 1, 2, 3, 4, 5, 6], engine='python',
                  skiprows=160000, nrows=1000)
    df_conv = list()
    avg_qber = 0.
    scaler.fit(df_whole.values)
    print("scaler parameters: max = {}, min = {}, scale = {}".format(scaler.data_max_, scaler.data_min_, scaler.scale_))
    avg = ExponentionalAverager(0.01, 5)
    block_qber = 0.
    ema_list = list()
    block_len = 25
    for index, row in df.iterrows():
        conv_row = row.tolist()
        avg_qber += conv_row[0]
        block_qber += conv_row[0]
        if index > 0 and index % block_len == 0:
            block_qber /= float(block_len)
            avg.add_value(block_qber)
            block_qber = 0.
        ema_list.append(avg.get_value())
        df_conv.append(conv_row)

    avg_qber /= len(df_conv)
    print("mean QBER = {}".format(avg_qber))
    deltas_est = list()
    deltas_predict = list()
    deltas_ema = list()
    real_qber_list = list()
    predict_qber_list = list()
    est_qber_list = list()
    model = load_lstm_model("../tr_model/model.json", "../tr_model/model.h5")
    i = 0
    avg_delay = 0
    buf = list()
    for test_row, real_res in zip(df_conv, df_conv[1:] + [df_conv[0]]):
        ema_val = ema_list[i]
        est_val = real_res[3]
        if len(buf) == 0:
            buf.append(test_row)
            buf.append(test_row)
            buf.append(test_row)
            buf.append(test_row)
        else:
            buf[0] = test_row
            shift(buf, 1)
        real_qber = real_res[0]
        i += 1
        start_ts = time.time()
        predict_res = predict_seq(buf, model)
        avg_delay += time.time() - start_ts
        deltas_est.append(abs(est_val - real_qber))
        deltas_predict.append(abs(real_qber - predict_res[0]))
        predict_qber_list.append(predict_res[0])
        real_qber_list.append(real_qber)
        est_qber_list.append(est_val)
        deltas_ema.append(abs(real_qber - ema_val))
    avg_delay /= (i + 1)
    plt.rcParams.update({'font.size': 22})
    sample_indices = [x for x in range(0, len(deltas_predict))]
    plt.plot(sample_indices, deltas_est, 'r', label=r'$\delta_{predict}$')
    plt.plot(sample_indices, deltas_predict, 'g', label=r'${\delta_{est}}$')
    plt.plot(sample_indices, deltas_ema, 'b', label=r'${\delta_{ema}}$')
    plt.grid()
    plt.figure()
    plt.plot(sample_indices, real_qber_list, 'salmon', label=r'$qber_{real}$')
    plt.plot(sample_indices, predict_qber_list, 'g', label=r'${qber_{predict}}$')
    plt.plot(sample_indices, est_qber_list, 'r', label=r'${qber_{est}}$')
    plt.grid()
    plt.show()
    model.save('keras_model.h5', include_optimizer=False)
    print("mean est score = {}".format(np.mean(deltas_est)))
    print("mean predict score = {}".format(np.mean(deltas_predict)))
    print("mean ema score = {}".format(np.mean(deltas_ema)))
    print("avg delay = {}ms".format(avg_delay * 1000))


if __name__ == '__main__':
    main()

