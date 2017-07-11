
# coding: utf-8

# In[118]:

import numpy as np
# import h5py
import scipy.io
np.random.seed(1337) # for reproducibility
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from seya.layers.recurrent import Bidirectional # after git clone need to python setup.py install again
# from keras.utils.layer_utils import print_layer_shapes



# In[119]:

print 'loading data'
# trainmat = h5py.File('/Users/jx/Downloads/deepsea_train/train.mat')
validmat = scipy.io.loadmat('validALLcontinuous.mat')
testmat = scipy.io.loadmat('testALLcontinuous.mat')

trainmat = scipy.io.loadmat('trainALLcontinuous.mat')


# In[120]:

X_train = np.transpose(np.array(trainmat['tr'][0][0][0]),axes=(0,2,1))
y_train = np.array(trainmat['tr'][0][0][1])
print X_train.shape
print y_train.shape


# In[16]:

y_train = y_train.reshape((-1, 1))


# In[121]:
lr = 1e-4#learning rate
reg = 1e-6
print 'building model'
nb_filters=32
model = Sequential()
model.add(Convolution1D(128, 4, border_mode='same', input_shape=(4, 113)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Convolution1D(256, 4, border_mode='same', input_shape=(4, 113)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution1D(512, 4, border_mode='same', input_shape=(4, 113)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# model.add(Convolution1D(64, 4))
# model.add(Activation('relu'))
# model.add(Convolution1D(64, 4))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_length=4))

# model.add(Dropout(0.2))

# model.add(forward_lstm)
# model.add(brnn)

# model.add(Dropout(0.2))
# model.add(backward_lstm)
# model.add(Dropout(0.2))

model.add(Flatten())

# model.add(Dense(input_dim=1, output_dim=1))
# model.add(Activation('relu'))

# model.add(Dense(input_dim=1, output_dim=1))
# model.add(LSTM(input_dim=30,output_dim=30,return_sequences=True))
# model.add(Dropout(0.2))

# model.add(Flatten())

# model.add(Dense(input_dim=78, output_dim=78))
# model.add(Activation('relu'))

model.add(Dense(1000))
model.add(Dense(1))
model.add(Activation('tanh'))
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


# In[122]:

print 'compiling model'
model.compile(loss='mse', optimizer='adam', metrics=["mse"])


# In[123]:

print 'running at most 60 epochs'

checkpointer = ModelCheckpoint(filepath="bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

model.fit(X_train, y_train, batch_size=256, nb_epoch=3, shuffle=True, validation_data=(np.transpose(validmat['v'][0][0][0],axes=(0,2,1)), validmat['v'][0][0][1]))
# model.fit(X_train, y_train, batch_size=100, nb_epoch=60, shuffle=True, show_accuracy=True, validation_data=(np.transpose(validmat['vsmall'][0][0][0],axes=(0,2,1)), validmat['vsmall'][0][0][1]), callbacks=[checkpointer,earlystopper])


X_test=np.transpose(testmat['tt'][0][0][0],axes=(0,2,1))
y_test=testmat['tt'][0][0][1]
out=model.predict(X_test, batch_size=512, verbose=1)
plt.plot(y_test,out,'ro')

# In[ ]:



