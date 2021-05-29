'''
Script for testing the models

'''

#import libraries
import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import h5py
import os
from os import path
import sys
import pickle
from tensorflow import keras
from models.Conv import Conv_Model
import csv


def test(args):
    save_dir = path.join("results",args.data_name)
    os.makedirs(save_dir,exist_ok=True)

    #load the data
    file_test = h5py.File(path.join(args.data_dir,args.data_name + ".hdf5"), 'r')
    x_train = file_test['x_train'][()]
    y_train = file_test['y_train'][()]
    x_test = file_test['x_test'][()]
    y_test = file_test['y_test'][()]
    file_test.close()

    in_shape = x_train.shape[1:]

    # determine the number of classes
    n_classes = len(np.unique(y_train))
    del x_train,y_train

    print('Train: X=%s, y=%s' % (x_test.shape, y_test.shape))

    #load the trained model
    #first, define the model
    model = Conv_Model(in_shape=in_shape,n_classes=n_classes)

    print(f'Loading weights from {args.checkpoint}')


    model.load_weights(args.checkpoint)

    print("--- Evaluation time ------")
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    with open(path.join(save_dir,"results.csv"), 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        
        writer.writerow([args.exp_name,acc]) #add more metrics
    










if __name__ == "__main__":

    '''
    run the script with:
    python test.py --data_name --data_dir --checkpoint --exp_name
    data_dir = "D:\download-D\Practice_Online_Courses\Hobby_Projects\Tf_project\data\mnist"
    exp_name = "exp_mnist_batch_128_epoch_2"
    checkpoint = "D:D:\download-D\Practice_Online_Courses\Hobby_Projects\Tf_project\logs\mnist\exp_mnist_batch_128_epoch_2\weights\checkpoint"
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", 
        help="The name of the dataset to use from tensorflow", default="mnist")
    
    parser.add_argument("--data_dir", help="Directory to load the test data")

    parser.add_argument("--checkpoint", help="Directory of the checkpoint")

    parser.add_argument("--exp_name", help = "Name of the experiment")
    

    args = parser.parse_args()
    test(args = args)