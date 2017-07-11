
# coding: utf-8

# In[9]:

import scipy.io
import numpy as np
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
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.optimizers import Adam, SGD

from keras import initializations

def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)



# In[10]:


validmat = scipy.io.loadmat('valid4D_mid16.mat')
testmat = scipy.io.loadmat('test4D_mid16.mat')
trainmat = scipy.io.loadmat('train4D_mid16.mat')


X_train = np.transpose(np.array(trainmat['tr'][0][0][0]),axes=(0,2,3,1))
y_train = np.array(trainmat['tr'][0][0][1]).squeeze()
# y_train = y_train.reshape((-1, 1))
# y_train.squeeze()
print X_train.shape
print y_train.shape



# In[11]:



lr = 1e-6#learning rate
reg = 1e-3
print 'building model'
nb_filters=32
model = Sequential()
# model.add(LSTM(32,  W_regularizer=l2(reg),return_sequences=True, input_shape=(4, 113)))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(LSTM(64,  W_regularizer=l2(reg),return_sequences=True)) # return sequences is needed for stacking
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(LSTM(128,  W_regularizer=l2(reg)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30))
model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]) ))

model.add(Dense(20,W_regularizer=l2(reg)))
model.add(Dropout(0.1))

model.add(Dense(10,W_regularizer=l2(reg)))
model.add(Dropout(0.1))

model.add(Dense(10,W_regularizer=l2(reg)))
model.add(Dropout(0.1))


# model.add(Dense(1))
# 
# model.add(Convolution2D(8, 2,12, border_mode='same', W_regularizer=l2(reg),init=my_init,input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))) # adding conv layer collapses output
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(4, 2,12, border_mode='same', W_regularizer=l2(reg),init=my_init)) # adding conv layer collapses output
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(4, 2,12, border_mode='same', W_regularizer=l2(reg),init=my_init)) # adding conv layer collapses output
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# 
# 



# model.add(Flatten())


# model.add(Dense(30))
model.add(Dense(1))
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


print 'compiling model'
model.compile(loss='mse', optimizer='adam', metrics=["mse"])



checkpointer = ModelCheckpoint(filepath="bestmodel4D_mid16redo.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=250, verbose=1)

X_valid=np.transpose(validmat['v'][0][0][0],axes=(0,2,3,1))
y_valid=np.array(validmat['v'][0][0][1]).squeeze()


model.fit(X_train, y_train, batch_size=256, nb_epoch=1000, shuffle=True, validation_data=(X_valid, y_valid),callbacks=[checkpointer,earlystopper])



# In[ ]:




# In[4]:

# 
# out=model.predict(X_train, batch_size=512,verbose=1)

model.summary()


# In[5]:

# get_ipython().magic(u'matplotlib')
import matplotlib.pyplot as plt

# plt.plot(y_train,out,'ro')


# In[6]:

# get_ipython().magic(u'matplotlib')
X_test=np.transpose(testmat['tt'][0][0][0],axes=(0,2,3,1))

y_test=np.array(testmat['tt'][0][0][1]).squeeze()

outtest=model.predict(X_test, batch_size=512,verbose=1)
plt.plot(y_test,outtest,'ro')
plt.show()


# In[ ]:



