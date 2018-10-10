__author__ = 'ZGolgooni'

import numpy as np
from Tools.file.read_list import load_file
from Tools.file.read_sample import read_sample
from Tools.ecg.r_peaks import detect_r_points, analyze_rpoints


def load_data_new(main_path, main_file, train_samples_id, dimension, step=2, train=False, preprocess=False):
    ids,paths, names, sampling_rates, labels, explanations, partitions,intervals = load_file(main_path, main_file)
    train_x = np.empty((0, dimension))
    train_y = np.empty((0, 2))
    for i in train_samples_id:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i],explanations[i],intervals[i], dimension=dimension, step=step, train=train, preprocess=preprocess)
        train_x = np.append(train_x, sample_x, axis=0)
        train_y = np.append(train_y, sample_y)
        #print('%d   label = %s, #sample = %d   %d' %(i,labels[i], sample_x.shape[0], sample_y.shape[0]))

    train_x = np.reshape(train_x, [train_x.shape[0], dimension, 1])
    train_y = np.reshape(train_y, [train_y.shape[0], 1])
    print('Total load result:  #sample0=%d, #sample1=%d' %(train_x.shape[0], train_y.shape[0]))
    return train_x, train_y


def load_sample(path, name, real_label, sampling_rate, explanation,interval, dimension=4000, step=2, train=False, preprocess=False):
    x_signal, y_signal = read_sample(path, name, sampling_rate, preprocess)
    length = y_signal.shape[0]
    rpeaks = detect_r_points(y_signal, length, sampling_rate)
    sample_x = np.empty([0, dimension])
    sample_y = np.empty([0, 1])
    if real_label == 'Normal':
        label = 0
    else:
        label = 1
    for index in range(0, len(rpeaks), step):
            if rpeaks[index] + dimension < length:
                if (train is False) | (real_label == 'Normal') | (explanation == 'Bradicardia') | (explanation == 'Tachicardia') | (explanation == 'Long QT'):
                    x = np.empty([dimension])
                    x = y_signal[rpeaks[index]:rpeaks[index] + dimension, 0]
                    y = np.array(label)
                    sample_x = np.append(sample_x, np.array([x]), axis=0)
                    sample_y = np.append(sample_y, np.array(label))
                elif check_overlap(interval,x_signal[rpeaks[index]],x_signal[rpeaks[index] + dimension]):
                    x = np.empty([dimension, 1])
                    x = y_signal[rpeaks[index]:rpeaks[index] + dimension, 0]
                    y = np.array(label)
                    sample_x = np.append(sample_x, np.array([x]), axis=0)
                    sample_y = np.append(sample_y, np.array(label))

    return sample_x, sample_y


def load_sample_test(path, name, sampling_rate=1000, dimension=5000, step=2):
    x_signal, y_signal = read_sample(path, name, sampling_rate)

    length = y_signal.shape[0]
    rpeaks = detect_r_points(y_signal, length, sampling_rate)
    sample_x = np.empty([0, dimension])
    for index in range(0, len(rpeaks), step):
            if rpeaks[index] + dimension < length:
                x = np.empty([dimension])
                x = y_signal[rpeaks[index]:rpeaks[index] + dimension, 0]
                sample_x = np.append(sample_x, np.array([x]), axis=0)
    return sample_x


def check_overlap(interval, min, max):
    if interval == '[:]':
        return True
    if interval == '':
        return False
    episodes = str.split(interval, ',')
    for e in episodes:
        a = str.split(e, ':')[0]
        b = str.split(e, ':')[1]
        if a == '[':
            start = 0
        else:
            start = float(a[1:])
        if b == ']':
            end = max + 1
        else:
            end = float(b[:len(b)-1])
        if get_overlap([min, max], [start, end]) > 0:
            return True

    return False


def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))
