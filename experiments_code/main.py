import cv2
import numpy as np
from sklearn.model_selection import train_test_split

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

# Models
from models import lstm_network, binary_network


def gpu_check():
    """
    return: True if gpu amount is greater than 1
    """
    return len(tf.config.experimental.list_physical_devices('GPU')) > 0

# I want to able to call this loss and other losses into function
# Where is the best place to put this function

# def loss(y_true, y_pred):
#     when_y_1 = y_true*tf.keras.backend.log(y_pred)*(1/weight_ratio)
#     neg_y_pred = Lambda(lambda x: -x)(y_pred)
#     when_y_0 = ( 1+Lambda(lambda x: -x)(y_true))*tf.keras.backend.log(1+neg_y_pred )

#     weighted_cross_entr = Lambda(lambda x: -x)(when_y_0+when_y_1)
#     return weighted_cross_entr

def lstm_train(traindict):
    """
    All this is doing is training the lstm network.
    After training make plots to see results. (Make a plotter class or functions)
    """
    train_x,train_y = norm_train_max_min(   data = traindict,
                                            max1=hyparams['max'],
                                            min1=hyparams['min']
                                        )
    train, val = {}, {}
    train['x'], val['x'],train['y'],val['y'] = train_test_split(    train_x,
                                                                    train_y,
                                                                    test_size = hyparams['val_size'])
    # used to do train test split in tensorify function
    # took out to make tensorify func universal
    train_data,val_data = tensorify(train, val)

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

    return model


def classifer_train(traindict, testdict, lstm):


    # I could have three main files that I load or
    # I could use an if statemet to switch bewteen the
    # experiments? What is the best method to use?
    # OR I could create another function that does loading?

    # Note that I am using the testing dable than Mac OS?
    # The answer is simple – more control to the user while providing betteta
    # If I do end up changing my normlization only need
    # to change in main file. Design intent

    # Test Data
    x,y = norm_train_max_min(   testdict,
                                max1 = hyparams['max'],
                                min1 = hyparams['min']
                                )

    iou = compute_iou(x,y,lstm)
    print('iou shape {}'.format(iou.shape) )
    # print('Intented quit')
    # quit()
    # Note that indices returned are in same order
    # as testdict unshuffled

    # Note that indexing works only if data is loaded the same way
    # Every time . Otherwise I could create an lstm model then I would train it.
    # If loaded again then I would need to make


    ## Find indices
    indices = return_indices(   testdict['abnormal'],
                                seed = hyparams['networks']['binary_classifier']['seed'],
                                abnormal_split = hyparams['networks']['binary_classifier']['abnormal_split']
                                )

    # Gets the train and test data
    # returns a dict with keys: x, y
    train, test = binary_data_split(iou, indices)


    if hyparams['exp_1']:
        if hyparams['exp_3']:
            # Makes same amount of normal and abnormal in train
            train = reduce_train(train['x'], train['y'])
            print(train['x'].shape)
            print(train['y'].shape)
           

        train, val = train_val_same_ratio(  train['x'],
                                            train['y'],
                                            val_ratio=hyparams['networks']['binary_classifier']['val_ratio']
                                            )

        print(train['x'].shape)
        print(train['y'].shape)

        print(val['x'].shape)
        print(val['y'].shape)
        print(train['y'][:10])
        print(np.bincount(train['y']))
        print(np.bincount(val['y']))
        print('For exp 1')
        # quit()

    else:
        x,y = norm_train_max_min(   traindict,
                                    max1 = hyparams['max'],
                                    min1 = hyparams['min']
                                    )
        iou = compute_iou(x, y ,lstm)


        # Note training_set_from_lstm has iou in first column
        # second colum is the index( Put in a filler -1)
        training_set_from_lstm = np.append( iou.reshape((-1,1)),
                                            -np.ones((len(iou), 1)),
                                            axis=1
                                            )
        temp_combined_train = {}

        print('train[x] {}'.format(train['x'].shape))
        print('training from lstm set {}'.format(training_set_from_lstm.shape))
        temp_combined_train['x']= np.append(    train['x'],
                                                training_set_from_lstm,
                                                axis=0
                                                )

        print('shape of tem[ combined from lstm set {}'.format(temp_combined_train['x'].shape))


        #since training set appended is coming from the orginal data
        # we know that all the labels are zeros because its normal

        temp_combined_train['y'] = np.append(train['y'],
                                                np.zeros(len(iou), dtype=np.int8)
                                                )

        if hyparams['exp_3']:
            temp_combined_train = reduce_train( temp_combined_train['x'],
                                                temp_combined_train['y']
                                                )

        train, val = train_val_same_ratio(  temp_combined_train['x'],
                                            temp_combined_train['y'],
                                            val_ratio=hyparams['networks']['binary_classifier']['val_ratio']
                                            )
        # print(train['x'].shape)
        # print(train['y'].shape)

        # print(val['x'].shape)
        # print(val['y'].shape)
        # print(train['y'][:10])
        # print(np.bincount(train['y']))
        # print(np.bincount(val['y']))
        # print('For exp 2')
        # quit()


    # Removing the index so I can pass into tensofify and not have two
    # functions that do similar things.
    train_no_index, val_no_index = {},{}
    val_no_index['x'] = val['x'][:,0]
    val_no_index['y'] = val['y']
 
    train_no_index['x'] = train['x'][:,0]
    train_no_index['y'] = train['y']


    print(train_no_index['x'].shape)
    print(train_no_index['y'].shape)
    print(val_no_index['x'].shape)
    print(val_no_index['y'].shape)

    print(np.sum(train_no_index['x']))
    print(train_no_index['x'][1400:1500])
    print(val_no_index['x'][1400:1500])
    print('Double check check here before moving on. Should only contain iou values')
    print('Intentful quit')
    # quit()
    train_tensor, val_tensor = tensorify(train_no_index, val_no_index)

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



    
    ## Stuck here Need to some how load the custom loss function into network func
    history, model= binary_network( train_tensor,
                                    val_tensor,
                                    model_loc=model_loc,
                                    nc =nc,
                                    weighted_binary=None, # filler
                                    output_bias =0, # Filler
                                    epochs=hyparams['epochs'],
                                    save_model=True
                                    )

    make_dir(loc['metrics_path_list'])
    plot_loc = join(    os.path.dirname(os.getcwd()),
                        *loc['metrics_path_list']
                        )
    accuracy_plot(history, plot_loc     )
    # go back and fix filler
    print('go back and fix filler')
    print('go fix how weight ratio is being added to')
    quit()
    # Need to save metric plots for classifer

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
    # returning model right now but might change that in future and load instead
    lstm_model = lstm_train(traindict)

    classifer_train(traindict, testdict, lstm_model)


    ## To_DO:
    """"
    fix why the y are not ints for classifer train and in abnormality
    will allow to use np.bincount

    Why do I need an initial bias for last layer. I think I got idea
    from google. But is it advantagous.
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

    1)Need to remove specifcation for GPU to run on


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
