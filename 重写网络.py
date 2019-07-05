from keras.models import Sequential, Model
from keras.utils.io_utils import HDF5Matrix
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, Dense, Reshape, concatenate,Conv1D,add,AveragePooling2D,multiply
from keras.layers import MaxPooling3D,Dropout,Flatten,Convolution3D,Lambda,MaxPooling2D,Activation
import numpy as np
from keras.utils import plot_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping
import scipy.io as scio
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import keras.backend as K
from keras.engine.topology import Layer
from keras.activations import softmax
from keras.utils import np_utils
import scipy.io as sio  
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
import random
import itertools
from pandas import DataFrame
import tensorflow as tf
import os
import h5py

def build_model10():
    global img_dim, output_shape
    
    input_layer = Input(shape=(32, 32, 10))
    
    conv1 = Conv2D(64,(3,3),strides=(3,3),activation='relu',kernel_initializer='uniform')(input_layer)
    conv21 = Conv2D(128,(3,3),strides=(3,3),activation='relu',kernel_initializer='uniform')(conv1)
    
    conv22 = Conv2D(128,(3,3),strides=(3,3),activation='relu',kernel_initializer='uniform')(conv1)
    
    
    mul1 = multiply([conv21,conv22])
    act = Activation('tanh')(mul1)
    flatten = Flatten()(act)
    
    den11 = Dense(128,activation='relu')(flatten)
    den12 = Dense(128,activation='relu')(flatten)
    mul2 = multiply([den11,den12])
    act2 = Activation('tanh')(mul2)
    
    output_layer = Dense(17,activation='softmax')(act2)
    
    train_model = Model(
        #inputs=input_layer,outputs=x1
        inputs=input_layer,outputs=output_layer
    )
    return train_model

train_model = build_model10()
train_model.summary()
train_model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=['accuracy'])

tensorboard =TensorBoard(log_dir='./graph')
filepath="./Result/md-{epoch:02d}-{acc:.5f}-{val_acc:.5f}.hdf5"
checkpoint = ModelCheckpoint(filepath = filepath, monitor='val_acc', mode='auto' ,save_best_only ='True')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience = 5, factor = 0.1 , min_lr = 0.0001)
callback_lists=[tensorboard, reduce_lr,checkpoint]
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
epochs = 150

train = h5py.File('./train/training.h5', 'r')
test = h5py.File('./test_a/round1_test_a_20181109.h5', 'r')
tr_sen1=train['sen1']
tr_sen2=train['sen2']
tr_label=train['label']
length=len(tr_sen1)

val = h5py.File('./validate/validation.h5', 'r')
val_sen1 = val['sen1']
val_sen2 = val['sen2']
x_val = np.concatenate((val_sen1, val_sen2), axis = 0)
val_label = val['label']

for i in range(1500):
    index = random.sample(range(length),200)
    index.sort()
    s1 = np.array(tr_sen1[index]) #.reshape((500,32,32,10))
    s2 = np.array(tr_sen2[index]) #.reshape((500,32,32,10))
    x = np.concatenate((s1, s2), axis = 0)
    y = np.array(tr_label[index]) #.reshape((500,17))
    train_model.fit(x, y, batch_size = 200, shuffle='batch', epochs = 1, 
                    validation_data = (x_val, val_label),
                    callbacks = callback_lists)
#train_model.load_weights('./Result/10-02-0.46223-0.59360.hdf5')
#train_model.fit(x = x_train10, y = y_train , batch_size = 200,shuffle='batch', epochs=epochs, validation_data=(x_val10, y_val),
#                 callbacks=callback_lists)

#
#train_model.load_weights('./Result/10-01-0.37325-0.62225.hdf5')
#y_pre = train_model.predict( [x_test8,x_test10])
#
#
#
#yy = np.argmax(y_pre,axis = 1)
#yyy=np_utils.to_categorical(yy,17).astype(int)
#df = DataFrame(yyy)
#df.to_csv('sub12.2no1.csv',index=False,header=None)