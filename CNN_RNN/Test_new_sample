__author__ = 'ZGolgooni'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, SimpleRNN, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU
import keras
import numpy as np
from CNN_RNN.prepare_data import load_sample_test
import plotly.plotly as py
import plotly.graph_objs as go
from Tools.file.read_sample import read_sample


"""
:param
file_path: file directory (e.g: '/myData/')
file_name: file name without .txt ('e.g: 'baseline')
sampling_rate: integer number, default = 1000
plot: determining that plotting is needed or not
"""
def check_new_input(file_path, file_name, sampling_rate=1000, plot=True):
    #Create and load networks
    dimension = 5000
    dimension_fraction = 1
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=16, kernel_size=80, strides=2, padding="same", input_shape=(dimension//dimension_fraction,dimension_fraction)))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('selu'))
    cnn_model.add(MaxPooling1D())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Conv1D(filters=16, kernel_size=80,padding="same"))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('selu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Conv1D(filters=16, kernel_size=80,padding="same"))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('selu'))
    cnn_model.add(MaxPooling1D())
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1))
    cnn_model.add(Activation('sigmoid'))
    adam = Adam(lr=0.0005)
    cnn_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    cnn_model.load_weights('./Trained_networks/cnn_model.h5')

    rnn_model = Sequential()
    init_one = keras.initializers.Ones()
    rnn_model.add(GRU(3, input_shape=(None, 1), kernel_initializer=init_one))
    rnn_model.add(BatchNormalization())
    rnn_model.add(PReLU())
    rnn_model.add(Dropout(0.5))
    rnn_model.add(Dense(1))
    rnn_model.add(Activation('sigmoid'))
    rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    rnn_model.load_weights('./Trained_networks/rnn_model.h5')


    #Classify new sample by CNN-RNN method
    sample_x = load_sample_test(file_path, file_name, sampling_rate=sampling_rate)
    if sample_x.shape[0] > 0:
        sample_x = np.reshape(sample_x, [sample_x.shape[0], dimension//dimension_fraction,dimension_fraction])
        
        step1_predicted = cnn_model.predict(np.array(sample_x))
        reshaped_predicted = np.reshape(step1_predicted, [1, step1_predicted.shape[0], step1_predicted.shape[1]])
        final_predicted = rnn_model.predict(reshaped_predicted)
        if final_predicted < 0.5:
            predicted_label = 'Normal'
        else:
            predicted_label = 'Arrhythmic'

    #Plot signal
    if plot == True:
        x_signal, y_signal =read_sample(file_path, file_name, sampling_rate=sampling_rate,preprocess=False)
        trace1 = go.Scatter(y=y_signal[:], x=x_signal[:], name='Signal')
        layout = go.Layout(title=file_name)
        figure = go.Figure(data=[trace1], layout=layout)
        py.plot(figure, filename=file_name)
        print('Plotting is done! :)  ')

    print('Predicted label = %s' %predicted_label)
    return predicted_label
