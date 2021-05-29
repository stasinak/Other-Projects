'''
This is the main script for training 
'''

import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import h5py
import os
from os import path
import sys
import pickle
from models.Conv import train_model
from utils.vis_utils import plot_history

# Main function for training
def train(args):

    exp_name = "exp_" + args.data_name + "_batch_" + str(args.batch_size) + "_epoch_" + str(args.max_epochs)
    
    #create directories for logs etc
    args.log_dir = path.join(args.log_dir,"logs",args.data_name, exp_name)
    os.makedirs(args.log_dir,exist_ok=True)


    file_writer = tf.summary.create_file_writer(path.join(args.log_dir , "metrics"))
    file_writer.set_as_default()
    

    #load the training data

    file_train = h5py.File(path.join(args.data_dir,args.data_name + ".hdf5"), 'r')
    x_train = file_train['x_train'][()]
    y_train = file_train['y_train'][()]
    file_train.close()


    in_shape = x_train.shape[1:]

    # determine the number of classes
    n_classes = len(np.unique(y_train))

    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))

    # ---------- define the model --------------------------------------
    
    
    #add the pars here
    history = train_model(data = [x_train, y_train], #data to used for training
                in_shape = in_shape, #shape of input
                n_classes = n_classes, #number of classes 
                log_dir =args.log_dir, #directory to store logs 
                val_prop = args.val_prop, #validation proportion
                max_epochs = args.max_epochs, #epochs
                batch_size = args.batch_size,#batch size
                exp_name = exp_name
                )

    #Add code to save the history as well!!!!!!
    with open(path.join(args.log_dir,'train_history'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    #add and save plot for training history
    # plot learning curves
    plot_history(history = history,
                log_dir = args.log_dir)
   
    

if __name__ == "__main__":
    
    '''
    run the script with:
    python train.py --data_name --data_dir --log_dir --max_epochs --batch_size --val_prop 
    
    data_dir = "D:\download-D\Practice_Online_Courses\Hobby_Projects\Tf_project\data\mnist"
    log_dir = "D:\download-D\Practice_Online_Courses\Hobby_Projects\Tf_project"
    max_epoch = 20
    batch_size = 128
    val_prop = 0.3
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", 
        help="The name of the dataset to use from tensorflow", default="mnist")
    
    parser.add_argument("--data_dir", help="Directory to load the data")

    parser.add_argument("--log_dir", help="Directory to save the model, logs etc")

    parser.add_argument("--max_epochs",help="Number of epochs to train", type = int)
    parser.add_argument("--batch_size",help="Batch size for training", type = int)
    parser.add_argument("--val_prop", help="Validation proportion during training", type= float)
    #parser.add_argument("--exp_name", help="Experiment name")

    args = parser.parse_args()
    train(args = args)


'''
    TO DO
    Add logs during training --> search how to store it properly
    Finish the test script --> Seems fine need to search it deeper
    Test script store acc in a csv file
    Train  
     
'''    