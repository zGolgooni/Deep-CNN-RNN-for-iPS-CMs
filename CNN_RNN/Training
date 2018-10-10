__author__ = 'ZGolgooni'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, Conv1D, MaxPooling1D, Flatten,SimpleRNN,AveragePooling1D
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
import keras
import numpy as np
from keras.activations import selu
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam,RMSprop
from Tools.file.read_list import load_file, split_samples, load_partitions
from Tools.file.train_test_fractions import load_fractions, load_train
from CNN_RNN.prepare_data import load_data_new, load_sample

num_experiments = 5

train_tp = []
train_tn = []
train_fp = []
train_fn = []
train_acc = []
train_sens = []
train_prec = []
test_tp = []
test_tn = []
test_fp = []
test_fn = []
test_acc = []
test_sens = []
test_prec = []

train_total = load_train()
train_fractions, validation_fractions = load_fractions()

main_path = './Data/'
main_file = 'Data_v960412.csv'
ids, paths, names, sampling_rates, labels, explanations,partitions,intervals = load_file(main_path, main_file)
##################################### Set parameters #####################################
dimension = 5000
dimension_fraction = 1
cnn_num_layer = 4
cnn_num_parameters = 0
batch_size = 32
epochs = 100
rnn_hidden_node = 3
rnn_dropout = 0.5
rnn_epochs = 100
rnn_batch_size  = 4
rnn_layer = 'GRU'  # LSTM #SimpleRNN #GRU
for run in range(0, num_experiments):
    train_samples_id = train_fractions[run]
    test_samples_id = validation_fractions[run]
##################################### Step 1 #####################################
    cnn_train_x, cnn_train_y = load_data_new(main_path, main_file, train_samples_id, dimension, train=True)
    cnn_train_x = np.reshape(cnn_train_x, [cnn_train_x.shape[0], dimension // dimension_fraction, dimension_fraction])
    cnn_test_x, cnn_test_y = load_data_new(main_path, main_file, test_samples_id, dimension, train=True)
    cnn_test_x = np.reshape(cnn_test_x, [cnn_test_x.shape[0], dimension // dimension_fraction, dimension_fraction])
    print('Build model...')
    #dimension = 5000
    #dimension_fraction = 1
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=16, kernel_size=80,strides=2,padding="same", input_shape=(dimension//dimension_fraction,dimension_fraction)))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('selu'))
    cnn_model.add(MaxPooling1D())
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Conv1D(filters=16, kernel_size=80,padding="same")) #, kernel_initializer='he_normal'
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('selu'))
    #cnn_model.add(MaxPooling1D())
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Conv1D(filters=16, kernel_size=80,padding="same"))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Activation('selu'))
    cnn_model.add(MaxPooling1D())
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Flatten())
    #cnn_model.add(Dense(5, activation='relu'))
    cnn_model.add(Dense(1))
    cnn_model.add(Activation('sigmoid'))
    adam = Adam(lr=0.0005)
    cnn_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
    cnn_history = cnn_model.fit(cnn_train_x, cnn_train_y, batch_size=batch_size, epochs=epochs, validation_split=0.12, callbacks=[reduce_lr])
##################################### Step 2 #####################################
    rnn_train_x = []
    rnn_train_y = []
    for i in train_samples_id:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], explanations[i],intervals[i], dimension=dimension,step=2, train=False)
        if sample_x.shape[0] > 0:
            sample_x = np.reshape(sample_x, [sample_x.shape[0], dimension // dimension_fraction, dimension_fraction])
            real_label = labels[i]
            if labels[i] == 'Normal':
                real_label = 0
            else:
                real_label =1
            sample_y = np.array([real_label])
            predicted = cnn_model.predict(np.array(sample_x))
            predicted = np.reshape(predicted,[predicted.shape[0]])
            rnn_train_x.append(predicted)
            rnn_train_y.append(sample_y)
    array_train_x = sequence.pad_sequences(rnn_train_x, maxlen=None, dtype='float64', padding='post', truncating='post',value=0.)
    array_train_x = np.reshape(array_train_x,[array_train_x.shape[0], array_train_x.shape[1],1])
    rnn_model = Sequential()
    init_one = keras.initializers.Ones()
    ##rnn_model.add(GRU(rnn_hidden_node, input_shape=(None, 1), kernel_initializer=init_one))
    rnn_model.add(LSTM(rnn_hidden_node, input_shape=(None, 1), kernel_initializer=init_one))
    #rnn_model.add(SimpleRNN(rnn_hidden_node, input_shape=(None, 1), kernel_initializer=init_one))

    rnn_model.add(BatchNormalization())
    rnn_model.add(PReLU())
    rnn_model.add(Dropout(rnn_dropout))
    rnn_model.add(Dense(1))
    rnn_model.add(Activation('sigmoid'))
    rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    array_train_x = sequence.pad_sequences(rnn_train_x, maxlen=None, dtype='float64', padding='post', truncating='post',
                                           value=0.)
    array_train_x = np.reshape(array_train_x, [array_train_x.shape[0], array_train_x.shape[1], 1])

    rnn_model.fit(array_train_x, np.array(rnn_train_y), batch_size=rnn_batch_size, nb_epoch=rnn_epochs,
                  validation_split=0.15)

##################################### Test on Test samples #####################################
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n = 0
    fp_samples = []
    fn_samples = []
    for i in test_samples_id:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], explanations[i],intervals[i], dimension=dimension,step=2, train=False)
        if sample_x.shape[0] > 0:
            sample_x = np.reshape(sample_x, [sample_x.shape[0], dimension//dimension_fraction,dimension_fraction])
            predicted = cnn_model.predict(np.array(sample_x))
            #for counter in range(0, predicted.shape[0]):
                #print('%d   %s  real label =%s    -> predicted = %f'%(i,names[i], labels[i], predicted[counter]))
            step1_predicted = cnn_model.predict(np.array(sample_x))
            reshaped_predicted = np.reshape(step1_predicted, [1, step1_predicted.shape[0], step1_predicted.shape[1]])
            final_predicted = rnn_model.predict(reshaped_predicted)
            if final_predicted < 0.5:
                predicted_label = 'Normal'
            else:
                predicted_label = 'Arrhythmic'
            #print('%d   %s  real label =%s    -> predicted = %s    %f'%(i,names[i], labels[i], predicted_label, final_predicted))
            n += 1
            if labels[i] == 'Normal':
                if predicted_label == 'Normal':
                    tn += 1
                else:
                    fp += 1
                    fp_samples.append(i)
            else:
                if predicted_label == 'Normal':
                    fn += 1
                    fn_samples.append(i)
                else:
                    tp += 1
    print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f, positive accuracy-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/n),(tp/(tp+fn))))
    test_tp.append(tp)
    test_tn.append(tn)
    test_fp.append(fp)
    test_fn.append(fn)
    test_acc.append((tp+tn)/n)
    test_sens.append(tp/(tp+fn))
    test_prec.append(tp/(tp+fp))
    print('+++++++++++++   (Test) Pay attention ++++++++++++++')
    print('fn:')
    print(fn_samples)
    print('fp:')
    print(fp_samples)
##################################### Test on Train samples #####################################
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n = 0
    fp_samples = []
    fn_samples = []
    for i in train_samples_id:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], explanations[i],intervals[i], dimension=dimension,step=2, train=False)
        if sample_x.shape[0] > 0:
            sample_x = np.reshape(sample_x, [sample_x.shape[0], dimension//dimension_fraction, dimension_fraction])
            #for counter in range(0, predicted.shape[0]):
                #print('%d   %s  real label =%s    -> predicted = %f'%(i,names[i], labels[i], predicted[counter]))
            step1_predicted = cnn_model.predict(np.array(sample_x))
            reshaped_predicted = np.reshape(step1_predicted, [1, step1_predicted.shape[0], step1_predicted.shape[1]])
            final_predicted = rnn_model.predict(reshaped_predicted)
            if final_predicted < 0.5:
                predicted_label = 'Normal'
            else:
                predicted_label = 'Arrhythmic'
            #print('%d   %s  real label =%s    -> predicted = %s    %f'%(i,names[i], labels[i], predicted_label, final_predicted))
            n += 1
            if labels[i] == 'Normal':
                if predicted_label == 'Normal':
                    tn += 1
                else:
                    fp += 1
                    fp_samples.append(i)
            else:
                if predicted_label == 'Normal':
                    fn += 1
                    fn_samples.append(i)
                else:
                    tp += 1
    print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f, positive accuracy-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/n),(tp/(tp+fn))))
    train_tp.append(tp)
    train_tn.append(tn)
    train_fp.append(fp)
    train_fn.append(fn)
    train_acc.append((tp + tn) / n)
    train_sens.append(tp / (tp + fn))
    train_prec.append(tp / (tp + fp))
    print('+++++++++++++   (Train) Pay attention ++++++++++++++')
    print('fn:')
    print(fn_samples)
    print('fp:')
    print(fp_samples)
##################################### Total result #####################################
tp = np.average(np.array(train_tp))
tn = np.average(np.array(train_tn))
fp = np.average(np.array(train_fp))
fn = np.average(np.array(train_fn))
train_accuracy = np.average(np.array(train_acc))
train_sensitivity = np.average(np.array(train_sens))
train_precision = np.average(np.array(train_prec))
print('**** Total : Train samples: ****')
print('\n--->Result for data = train , samples (%d Arrhythmic, %d Normal)' % ((fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, Accuracy-> %f, recall-> %f,  precision-> %f\n\n' % (tp, tn, fp, fn, train_accuracy,train_sensitivity,train_precision))
tp = np.average(np.array(test_tp))
tn = np.average(np.array(test_tn))
fp = np.average(np.array(test_fp))
fn = np.average(np.array(test_fn))
test_accuracy = np.average(np.array(test_acc))
test_sensitivity = np.average(np.array(test_sens))
test_precision = np.average(np.array(test_prec))
print('**** Total : Test samples: ****')
print('\n--->Result for data = test , samples (%d Arrhythmic, %d Normal)' % ((fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, Accuracy-> %f, recall-> %f,  precision-> %f\n\n' % (tp, tn, fp, fn, test_accuracy,test_sensitivity,test_precision))
##################################### Save results & parameters in file #####################################

for i in test_acc:
    print('test   %f, ' %i)
print('\n')
for i in train_acc:
    print('train %f, ' %i)

cnn_model.summary()
rnn_model.summary()
#cnn_model.get_config()
#rnn_model.get_config()
cnn_model.save_weights('cnn_model.h5')
rnn_model.save_weights('rnn_model.h5')
