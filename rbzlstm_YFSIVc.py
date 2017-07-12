from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import args
args = args.args
import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D,Convolution2D, MaxPooling1D
from keras.layers import merge
# from keras.layers.merge import Concatenate
from keras.regularizers import l2 #, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam, SGD
import keras
print(keras.__version__)
from scipy import stats

import numpy as np
import h5py
import scipy.io
validmat = scipy.io.loadmat('sTRSVdup_valid.mat')
testmat = scipy.io.loadmat('sTRSVdup_valid.mat')
trainmat = scipy.io.loadmat('sTRSVdup_train.mat')


X_train = np.transpose(np.array(trainmat['tr'][0][0][0]),axes=(0,2,1))
y_train = np.array(trainmat['tr'][0][0][1]).squeeze()

print(X_train.shape)
print(y_train.shape)



lr = args.learning_rate#learning rate
reg = args.reg
dropout = args.n_layers
n_layers = args.n_layers
n_LSTM = args.n_LSTM
n_units = args.n_units
n_LSTM_units = args.n_LSTM_units
epochs = args.epochs
batch_size = args.batch_size
name = args.name

print('building model')
nb_filters=32
model = Sequential()
for i in range(n_layers):
    model.add(Dense(n_units, input_shape=(4,113)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
for i in range(n_LSTM):
    model.add(LSTM(n_LSTM_units,  W_regularizer=l2(reg),return_sequences=True))#, input_shape=(100,113)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

#model.add(LSTM(512,  W_regularizer=l2(reg),return_sequences=True))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
# model.add(LSTM(128,  W_regularizer=l2(reg),return_sequences=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(LSTM(64,  W_regularizer=l2(reg),return_sequences=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1))


adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# sgd = SGD(lr=lr, decay=1e-6, momentum=0.09, nesterov=True)
print('compiling model')
model.compile(loss='mse', optimizer='adam', metrics=["mse"])


checkpointer = ModelCheckpoint(filepath="bestmodel3D.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

X_valid=np.transpose(validmat['tr'][0][0][0],axes=(0,2,1))
# X_valid=np.expand_dims(X_valid)
y_valid=np.array(validmat['tr'][0][0][1]).squeeze()


history=model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, shuffle=True, validation_data=(X_valid, y_valid),callbacks=[checkpointer, earlystopper])



out=model.predict(X_train, batch_size=512,verbose=1)
plt.figure(1)
plt.plot(y_train,out,'ro')
plt.savefig('%s_train.png'%name)



import scipy
out=out.squeeze()
slope, intercept, r_value, p_value, std_err = stats.linregress(y_train, out)
print(name, "r = ",r_value)
print(name, "r^2 = ",r_value**2)


X_test=np.transpose(testmat['tr'][0][0][0],axes=(0,2,1))
y_test=np.array(testmat['tr'][0][0][1]).squeeze()
y_out=model.predict(X_test, batch_size=512,verbose=1)
plt.figure(2)
plt.plot(y_test,y_out,'ro')
plt.ylabel('model prediction')
plt.xlabel('FACS-seq data')
plt.savefig('%s_test.png'%name)
model.summary()

import scipy
out=y_out.squeeze()
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, out)
print(name, "r = ",r_value)
print(name, "r^2 = ",r_value**2)
# print slope
# print intercept


plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('%s_loss.png'%name)






