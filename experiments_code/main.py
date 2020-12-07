import seaborn as sns
import cv2
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow import keras
import os, sys, time
from os.path import join


# Is hyperparameters and saving files config file
from config import hyparams, loc
# Design intent
# But hyparas as constants that can be called anywhere
# I only want to call loc here and not anywhere else

## Might delete havent used yet
from custom_metrics import bb_intersection_over_union, bb_intersection_over_union_np
from coordinate_change import xywh_tlbr, tlbr_xywh

# Data Info
from data import data_lstm, tensorify
from load_data import norm_train_max_min
from load_data_binary import *

# Plots
from metrics_plot import loss_plot

from models import lstm_network, binary_network


def gpu_check():
    """
    return: True if gpu amount is greater than 1
    """
    return len(tf.config.experimental.list_physical_devices('GPU')) > 0

# I want to able to call this loss and other losses into function
# Where is the best place to put this function

def loss(y_true, y_pred):
    when_y_1 = y_true*tf.keras.backend.log(y_pred)*(1/weight_ratio)
    neg_y_pred = Lambda(lambda x: -x)(y_pred)
    when_y_0 = ( 1+Lambda(lambda x: -x)(y_true))*tf.keras.backend.log(1+neg_y_pred )
    
    weighted_cross_entr = Lambda(lambda x: -x)(when_y_0+when_y_1)
    return weighted_cross_entr

def lstm_train(traindict):
    """
    All this is doing is training the lstm network.
    After training make plots to see results. (Make a plotter class or functions)
    """
    train_x,train_y = norm_train_max_min(   data = traindict,
                                            max1=hyparams['max'],
                                            min1=hyparams['min']
                                            )

    train_data,val_data = tensorify(train_x, train_y)

    #naming convection 
    nc = [  loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['frames']
            ] # Note that frames is the sequence input


    make_dir(loc['model_path_list']) # Make directory to save model

    # print(loc['model_path_list'])
    model_loc = join(   os.path.dirname(os.getcwd()), 
                        *loc['model_path_list']
                        ) # create save link


    history, model = lstm_network(  train_data,
                                    val_data, 
                                    model_loc=model_loc, ### Fix this line
                                    nc = nc, 
                                    epochs=hyparams['epochs']
                                    )


    make_dir(loc['metrics_path_list'])
    plot_loc = join(    os.path.dirname(os.getcwd()), 
                        *loc['metrics_path_list']
                        ) 
    # Note that loss plot is saved to plot_lcc
    loss_plot(history, plot_loc)
    

def classifer_train(testdict, lstm):

    # Note that I am using the testing data
    # If I do end up changing my normlization only need 
    # to change in main file. Design intent
    x,y = norm_train_max_min(   testdict,
                                max1 = hyparams['max'],
                                min1 = hyparams['min']
                                )

    iou = compute_iou(x,y,lstm)
    # Note that indices returned are in same order
    # as testdict unshuffled 
    indices = return_indices(   testdict['abnormal'],
                                seed = hyparams['seed'],
                                abnormal_split = hyparams['binary_classifier']['abnormal_split'])
    
    # returns a dict with keys: train_x, train_y, test_x, test_y
    data = binary_data_split(iou, indices)



    #naming convection 
    nc = [  loc['nc']['model_name_binary_classifer'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['frames']
            ] # Note that frames is the sequence input


    make_dir(loc['model_path_list']) # Make directory to save model

    # print(loc['model_path_list'])
    model_loc = join(   os.path.dirname(os.getcwd()), 
                        *loc['model_path_list']
                        ) # create save link

    binary_network( train_bm,
                    val_bm, 
                    model_loc=model_loc,
                    nc =nc,
                    weighted_binary,
                    output_bias,
                    epochs=hyparams['epochs'],
                    save_model=True):
    pass

def make_dir(dir_list):
    try:
        print(os.makedirs(join( os.path.dirname(os.getcwd()),
                                *dir_list )) )  
    except OSError:
        print('Creation of the directory {} failed'.format( join(os.path.dirname(os.getcwd()),
                                                            *dir_list) ) )
    else:
        print('Successfully created the directory {}'.format(   join(os.path.dirname(os.getcwd()),
                                                                *dir_list) ) )



def main():

    traindict, testdict = data_lstm(    loc['data_load']['avenue']['train_file'],
                                        loc['data_load']['avenue']['test_file']
                                        )
    lstm_train(traindict)


    ## To_DO:
    """"
    Need to remove specifcation for GPU to run on
    1)  figuring out name and folder creation 
        seems like I will want to save model, results plots etc into 
        a single folder so if I don't like the results I can delete
        the folder eniterly

    2)  Looks like I can save the hyprmas file as well in a txt file.
        Might be useful might not be 
    
    3)  Make a function or class that plots data and saves it. I talking about 
        the plots that I need to make for metrics.

    4) create a testing lstm. This would mean I need to return the model
        of lstm_train. Or more robustly just read saved model instead.
        Problem is if I'm running a test where don't want to save anything 
        how do I do that. Maybe move them to tmp

    """



if __name__ == '__main__':
    print('GPU is on: {}'.format(gpu_check() ) )


    main()

    print('Done')

  