# QBER-LSTM-based-estimation
Simple QBER forecast model based on LSTM RNN.
Depends on:
- tensorflow;
- numpy;
- matplotlib;
- pandas.

Folder tr_model contains weights for model trained on real dataset for our industrial QKD setup during generation process.
To review results launch:
`./lstm_live_forecast.py`. Graph will show QBER forecasts based on the LSTM prediction, physical model, and EMA filter.