'''
Pre processing module to prepare the data
'''

#import libraries
import tensorflow as tf
from matplotlib import pyplot as plt
import argparse
import numpy as np
import os
from os import path
import sys
import h5py







def pre_process(args):

    #create a directory for storing the data
    save_path = path.join(args.data_dir,"data",args.data_name)

    os.makedirs(save_path, exist_ok=True) 

    #load the dataset
    #Here create another script for splitting and pre-processing the data!
    dataset = eval("tf.keras.datasets."+args.data_name)
    

    (x_train,y_train), (x_test,y_test) = dataset.load_data()
    x_train,x_test = x_train/255, x_test/255

    #print the shape of the data
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

    


    # --------- pre processing for the images ------------------------
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    in_shape = x_train.shape[1:]

    # determine the number of classes
    n_classes = len(np.unique(y_train))
    

    # normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    for i in range(25):

        # define subplot
        plt.subplot(5, 5, i+1)
        # plot raw pixel data
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))


    plt.savefig(path.join(save_path, 'data_sample.png'))

    #write the dataset
    with h5py.File(path.join(save_path,args.data_name+".hdf5"), "w") as f:
        f.create_dataset("x_train",data =  x_train)
        f.create_dataset("y_train", data =y_train)
        f.create_dataset("x_test", data =x_test)
        f.create_dataset("y_test", data =y_test)
        
    print("---- You are done with pre processing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", 
        help="The name of the dataset to use from tensorflow", default="mnist")
    
    parser.add_argument("--data_dir", help="Directory to store data")

    

    args = parser.parse_args()
    pre_process(args = args)


