'''
This script is used for training 

'''

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
from os import path



def Conv_Model(in_shape, #shape of input
                n_classes, #number of classes
                ):


    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    
    

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(data,
                in_shape, #shape of input
                n_classes, #number of classes
                log_dir, #directory to store logs 
                val_prop, #validation proportion
                max_epochs, #epochs
                batch_size, #batch size,
                exp_name #expirement name
                ): 

    x_train = data[0]
    y_train = data[1]

    model = Conv_Model(in_shape,n_classes)
    print(model.summary())
    #Add more callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=10)
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    cp_callback = ModelCheckpoint(filepath=path.join(log_dir,"weights",
                                             "checkpoint--{ep:02d}--{val_acc:.2f}.hdf5"),
                                                 verbose=1)


    history = model.fit(x_train, y_train, 
                        validation_split = val_prop,
                        epochs=max_epochs,     
                        batch_size= batch_size,
                        callbacks = [early_stop, tensorboard_callback,cp_callback])


    model.save(path.join(log_dir, "model.h5")) #save the model

    print(" -------- Training is done --------")
    return history
