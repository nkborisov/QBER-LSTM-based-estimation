import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, Activation, Input, Masking
from tensorflow.keras.layers import LSTM
from keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        b = dataset[i + look_back, 0]
        dataX.append(a)
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)


def load_lstm_model(json_path, weights_path):
    print(tf.keras.backend.image_data_format())
    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Model loaded from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    return loaded_model


def plot_loss(history):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    #plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    plt.show()


def learn(trainX, trainY, look_back, n_features):
    # create and fit the LSTM network
    batch_size = 2048
    s = trainX.shape[0]
    t = trainX.shape[1]
    model = Sequential()
    # model.add(Input(batch_input_shape=(batch_size, look_back, n_features)))
    model.add(Bidirectional(LSTM(units=128, return_sequences=False, stateful=False, input_shape=(look_back, n_features),
                                 go_backwards=True)))
    # model.add(Dropout(0.05))
    model.add(Dense(n_features))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(trainX, trainY, epochs=200, batch_size=batch_size, validation_split=0.1, verbose=2)
    print(model.summary())
    # Plot training
    # evaluate the model
    scores = model.evaluate(trainX, trainY, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # serialize model to  JSON
    model_json = model.to_json()
    with open("../tr_model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../tr_model/model.h5")
    # print("Saved model to disk")
    plot_loss(history)
    return model


def main():
    # fix random seed for reproducibility
    fixed_seed = 7
    look_back = 4
    n_features = 7

    numpy.random.seed(fixed_seed)
    # load the dataset
    pulses_stats_file_path = "../dataset/fr_gains.csv"
    dataframe = read_csv(pulses_stats_file_path, usecols=[0, 1, 2, 3, 4, 5, 6], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    print("Training set size = {}, testing set size = {}".format(train_size, test_size))
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, n_features))
    testX = numpy.reshape(testX, (testX.shape[0], look_back, n_features))
    # make predictions
    model = load_lstm_model("../tr_model/model.json", "../tr_model/model.h5") #
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Get something which has as many features as dataset
    trainPredict_extended = numpy.zeros((len(trainPredict), n_features))
    # Put the predictions there
    trainPredict_extended[:, 0] = trainPredict[:, 0]
    # Inverse transform it and select the 0th column.
    trainPredict = scaler.inverse_transform(trainPredict_extended)[:, 0]
    # Get something which has as many features as dataset
    testPredict_extended = numpy.zeros((len(testPredict), n_features))
    # Put the predictions there
    testPredict_extended[:, 0] = testPredict[:, 0]
    # Inverse transform it and select the 3rd column.
    testPredict = scaler.inverse_transform(testPredict_extended)[:, 0]

    trainY_extended = numpy.zeros((len(trainY), n_features))
    trainY_extended[:, 0] = trainY
    trainY = scaler.inverse_transform(trainY_extended)[:, 0]

    testY_extended = numpy.zeros((len(testY), n_features))
    testY_extended[:, 0] = testY
    testY = scaler.inverse_transform(testY_extended)[:, 0]

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.4f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.4f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, 0] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1: len(dataset) - 1, 0] = testPredict

    # plot
    e_mu_act = scaler.inverse_transform(dataset)[:, 0]
    series, = plt.plot(e_mu_act)
    test_prediction = testPredictPlot[:, 0]
    prediction_training, = plt.plot(trainPredictPlot[:, 0], linestyle='--')
    prediction_test, = plt.plot(test_prediction, linestyle='--')
    plt.title(r'$E_\mu$ LTSM prediction experiment')
    plt.ylabel(r'$E_\mu$')
    plt.xlabel('Frame number')
    plt.grid()
    plt.show()

    plt.figure()
    frag_beg = 86250
    frag_end = 87250
    frame_indices = [i for i in range(frag_beg, frag_end)]
    plt.plot(frame_indices, e_mu_act[frag_beg:frag_end], label='actual', color='b')
    plt.plot(frame_indices, test_prediction[frag_beg:frag_end], label='prediction', color='r')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
