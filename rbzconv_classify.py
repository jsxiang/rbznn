
# coding: utf-8

# In[13]:

import numpy as np
import scipy.io
# import h5py
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D,Convolution2D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.optimizers import Adam, SGD

from keras import initializations

# In[14]:

def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.001, name=name)


#Load data directly from file
f = open('goodmusBinary.txt', 'r')
y = []
for line in f:
	y.append(float(line))

f = open('goodseqsBinary.txt', 'r')
seqs = []
for line in f:
	seqs.append(line.strip('\n'))

# ATGC to one hot encoding
seq_map = {'A': np.array([1,0,0,0]), 'T': np.array([0,1,0,0]), 'G': np.array([0,0,1,0]), 'C': np.array([0,0,0,1])}

#sequences to matrix form
X = []
for seq in seqs:
	X.append(np.vstack([seq_map[s] for s in seq]))

# zero pad sequences (end)
max_len = 0
for s in X:
	if len(s)>max_len:
		max_len = len(s)
for i in range(len(X)):
	p = max_len - X[i].shape[0]
	if p>0:
		X[i] = np.vstack([X[i], np.zeros((p,4))])

X= np.asarray(X)
y = np.asarray(y)
print X.shape

#shuffle indices...not data!
idx = range(len(X))
np.random.shuffle(idx)
train_idx = idx[:int(len(X)*.85)]
test_idx = idx[int(len(X)*.85):]

#split into train/test
X_train = X[train_idx]
X_test = X[int(len(X)*.85):]
y_train = y[:int(len(X)*.85)]
y_test = y[int(len(X)*.85):]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[2], 1, X_train.shape[1])) #(N, F, H, W)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[2], 1, X_test.shape[1])) #(N, F, H, W)

print X_train.shape
# print 'loading data'
# validmat = scipy.io.loadmat('validALL.mat')
# testmat = scipy.io.loadmat('testALL.mat')

# trainmat = scipy.io.loadmat('trainALL.mat')


# # In[15]:

# X_train = np.transpose(np.array(trainmat['tr'][0][0][0]),axes=(0,2,1))
# y_train = np.array(trainmat['tr'][0][0][1]).squeeze()

# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, X_train.shape[2])) #(N, F, H, W)
# print X_train.shape
# print y_train.shape


# In[16]:
lr = 1e-5#learning rate
reg = 1e-6

print 'building model'
model = Sequential()
model.add(Convolution2D(128, 4, 4, border_mode='same', input_shape=(X_train.shape[1], 1, X_train.shape[3]), W_regularizer=l2(reg), init=my_init))
model.add(Activation('relu'))
model.add(Convolution2D(256, 4, 4, border_mode='same', W_regularizer=l2(reg), init=my_init))
model.add(Activation('relu'))
model.add(Convolution2D(512, 4, 4, border_mode='same', W_regularizer=l2(reg), init=my_init))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(1000))
model.add(Dense(1))
model.add(Activation('tanh')) #[-1, 1]


# In[17]:
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# sgd = SGD(lr=lr, momentum=0.9)
# model.compile(loss='mse',
#               optimizer='adam', metrics=['mse'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])


print 'compiling model'
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["mse"])

# In[18]:

# print 'running at most 60 epochs'

# checkpointer = ModelCheckpoint(filepath="bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
model.fit(X_train, y_train,nb_epoch=1000, batch_size=1024, verbose=1, show_accuracy=True, validation_data=(X_test, y_test),callbacks=[earlystopper])

# model.fit(X_train, y_train, batch_size=256, nb_epoch=60,  verbose=1, show_accuracy=True, shuffle=True, validation_data=(X_test, y_test))#, callbacks=[checkpointer,earlystopper])


# In[10]:

model.summary()


# In[ ]:



