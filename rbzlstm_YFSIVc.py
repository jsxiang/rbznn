
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
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam, SGD
import keras
print keras.__version__



import numpy as np
import h5py
import scipy.io
validmat = scipy.io.loadmat('sTRSVdup_valid.mat')
testmat = scipy.io.loadmat('sTRSVdup_valid.mat')
trainmat = scipy.io.loadmat('sTRSVdup_train.mat')


X_train = np.transpose(np.array(trainmat['tr'][0][0][0]),axes=(0,2,1))
y_train = np.array(trainmat['tr'][0][0][1]).squeeze()

print X_train.shape
print y_train.shape



lr = 1e-5#learning rate
reg = 1e-3
print 'building model'
nb_filters=32
model = Sequential()
model.add(LSTM(128,  W_regularizer=l2(reg),return_sequences=True, input_shape=(4,113)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(256,  W_regularizer=l2(reg),return_sequences=True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
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
print 'compiling model'
model.compile(loss='mse', optimizer='adam', metrics=["mse"])


checkpointer = ModelCheckpoint(filepath="bestmodel3D.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

X_valid=np.transpose(validmat['tr'][0][0][0],axes=(0,2,1))
# X_valid=np.expand_dims(X_valid)
y_valid=np.array(validmat['tr'][0][0][1]).squeeze()


history=model.fit(X_train, y_train, batch_size=512, nb_epoch=500, shuffle=True, validation_data=(X_valid, y_valid),callbacks=[checkpointer,earlystopper])



out=model.predict(X_train, batch_size=512,verbose=1)
plt.figure(1)
plt.plot(y_train,out,'ro')




import scipy
out=out.squeeze()
print out.shape
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_train, out)
print "r = ",r_value
print "r^2 = ",r_value**2


X_test=np.transpose(testmat['tr'][0][0][0],axes=(0,2,1))
y_test=np.array(testmat['tr'][0][0][1]).squeeze()
y_out=model.predict(X_test, batch_size=512,verbose=1)
plt.figure(2)
plt.plot(y_test,y_out,'ro')
plt.ylabel('model prediction')
plt.xlabel('FACS-seq data')

model.summary()

import scipy
out=y_out.squeeze()
print out.shape
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_test, out)
print "r = ",r_value
print "r^2 = ",r_value**2
# print slope
# print intercept


plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()






