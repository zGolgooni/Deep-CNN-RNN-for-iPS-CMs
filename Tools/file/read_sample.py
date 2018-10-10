__author__ = 'Zeynab'
import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
from biosppy.signals.tools import smoother
from scipy.signal import decimate
from scipy.stats import zscore
from pywt import dwt,idwt


#Set parameters
min_range = -50
max_range = 50
rate = 1000
total_length = 60


'''
read sample
Normalize data between (-50,50)
Downsample to rate (if sampling rate is different)
If true signal will be simply preprocessd(smoothed by biosppy tool)
'''

#Current main function to read a new sample(.txt file)
def read_sample(path, name, sampling_rate, preprocess=True):
    dataset = pandas.read_csv(path + name + '.txt', delimiter='\t', skiprows=4, skipfooter=1,engine='python')
    x_signal = dataset.values[:total_length * sampling_rate, 0]
    y = dataset.values[:total_length * sampling_rate, 1]

    y_signal = normalize_data(pandas.DataFrame(y), max_range, min_range)

    if preprocess is True:
        smoothed_signal, params = smoother(y_signal[:,0])
        y_signal = np.reshape(smoothed_signal, [smoothed_signal.shape[0],1])

    if sampling_rate != rate:
        down_sample_factor = int(sampling_rate // rate)
        y_signal = decimate(y_signal, down_sample_factor)
        #x_signal = decimate(x_signal, down_sample_factor)
        y_signal = normalize_data(pandas.DataFrame(y), max_range, min_range)

    return x_signal, y_signal


#Normalize data in specified range
def normalize_data(dataset, max=max_range, min=min_range):
    scaler = MinMaxScaler(feature_range=(min, max))
    data = scaler.fit_transform(dataset)
    #move to fit baseline to zero
    index = np.where(dataset == 0)
    value = data[index[0][0]]
    data = data-value
    return data
