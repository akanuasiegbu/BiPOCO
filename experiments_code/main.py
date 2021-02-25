import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow import keras
import os, sys, time
from os.path import join


# Is hyperparameters and saving files config file
from config import hyparams, loc, exp
# Design intent
# But hyparas as constants that can be called anywhere
# I only want to call loc here and not anywhere else

## Might delete havent used yet
from custom_metrics import bb_intersection_over_union, bb_intersection_over_union_np
from coordinate_change import xywh_tlbr, tlbr_xywh
from TP_TN_FP_FN import *

# Data Info
from data import data_lstm, tensorify, data_binary
from load_data import norm_train_max_min
from load_data_binary import compute_iou
# Plots
from metrics_plot import *

# Models
from models import lstm_network, binary_network
import wandb

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

def iou_as_probability(testdict, model):
    """
    Note that to make abnormal definition similar to orgininal definition
    Need to switch ious because low iou indicates abnormal and high 
    iou indicate normal
    testdict: 
    model: traj prediction model
    iou_prob: probability where high indicates abnormal pedestrain and low
              indicates normal pedestrain
    """ 
    # model here is lstm model
    # need to normalize because lstm expects normalized
    x,y = norm_train_max_min(   testdict,
                                # max1 = hyparams['max'],
                                # min1 = hyparams['min']
                                max1 = max1,
                                min1 = min1
                                )

    iou = compute_iou(x, y, model)
    iou_prob =  1 - iou
    return  iou_prob

def ped_auc_to_frame_auc_data(model, testdict, test_bin=None):
    """
    Note that this does not explictly calcuate frame auc
    but removes select points to reduce to frame AUC data.

    # To do 
    Function will need to be appended to add the 3_2 method.
    When normal data can be drawn from the training set.
    Would expect percentage of data removed to decrease
    # To do

    testdict: From orginal data test dict
    test_bin: binary classifer test dict
    model: binary classifer model or lstm model


    return:
    test_auc_frame: people that are in the frame as proability and index of frame
    remove_list: indices of pedestrain removed
    """
    if not test_bin:
        # calc iou prob
        iou_prob = iou_as_probability(testdict, model)
        
        # since test dict was not shuffled lets created index
        test_index = np.arange(0, len(testdict['abnormal']), 1)
        
        vid_loc = testdict['video_file'].reshape(-1,1) #videos locations
        frame_loc = testdict['frame_y'].reshape(-1,1) # frame locations
    
    else:
        test_bin_index = test_bin['x'][:,1]

        test_bin_index = test_bin_index.astype(int) 

        vid_loc = testdict['video_file'][test_bin_index].reshape(-1,1) #videos locations
        frame_loc = testdict['frame_y'][test_bin_index].reshape(-1,1) # frame locations
    
    # encoding video and the frame together 
    vid_frame = np.append(vid_loc, frame_loc, axis=1)


    
    # Treating each row vector as a unique element 
    # and looking for reapeats
    unique, unique_inverse, unique_counts = np.unique(vid_frame, axis=0, return_inverse=True, return_counts=True)


    #  finds where repeats happened and gives id for input
    # into unique_inverse
    
    repeat_inverse_id = np.where(unique_counts>1)[0]
    
    # Pedestrain AUC equals Frame AUC
    if len(repeat_inverse_id) == 0:
        # print('Ped AUC = Frame AUC')
        if not test_bin:
            test_auc_frame = 'not possible'
        else: 
            test_auc_frame = test_bin

    # Convert Pedestrain AUC to Frame AUC
    else:
        # print('Ped AUC != Frame AUC')
        # print(repeat_inverse_id.shape)
        # # find pairs given repeat_inverse_id
        remove_list_temp = []
        for i in repeat_inverse_id:
            # find all same vid and frame
            if not test_bin:
                same_vid_frame = np.where(unique_inverse == i)[0]
                # Note that this is treating iou_as_prob
                y_pred = iou_prob[same_vid_frame]
                # if sum(y_pred) == 0:
                #     rand = np.random.randint(len(y_pred))
                #     index = np.arange(0,len(y_pred) )
                #     temp = np.delete(index, rand)
                #     # would need to check for if len(y_pred )==1
                    
                # else:
                temp = np.where(y_pred != np.max(y_pred))[0]
                
                remove_list_temp.append(same_vid_frame[temp])

            else:
                same_vid_frame = np.where(unique_inverse == i )[0]
                # print('same_vid_frame {}'.format(same_vid_frame))
                # print('vid_frame :{}'.format(vid_frame[same_vid_frame]))

                y_pred = model.predict(test_bin['x'][:,0][same_vid_frame])
                # find max y_pred input other indices to remove list
                
                temp = np.where(y_pred != np.max(y_pred))[0]

                remove_list_temp.append(same_vid_frame[temp])
                
            # print('y_pred :{}'.format(y_pred))
            # print('removed elements {}'.format(temp))
            # print('*'*20)
            # print('\n')
        
        
        remove_list = [item for sub_list in remove_list_temp for item in sub_list]

        # print(np.array(flat_list).shape) #shape matches pairs + extra pairs if (pair>2)
             
        remove_list = np.array(remove_list).astype(int)

        # print('Length of removed elements is :{}'.format(len(remove_list)))
        # print(test_bin['x'].shape)
        test_auc_frame = {}
        if not test_bin:
            iou_prob_per_frame = np.append(iou_prob.reshape(-1,1), test_index.reshape(-1,1), axis=1)
            test_auc_frame['x'] = np.delete( iou_prob_per_frame, remove_list, axis = 0 )
            test_auc_frame['y'] = np.delete(testdict['abnormal'].reshape(-1,1) , remove_list, axis=0)
        else:
            test_auc_frame['x'] = np.delete(test_bin['x'], remove_list, axis=0)
            test_auc_frame['y'] = np.delete(test_bin['y'], remove_list, axis=0)
        # print(test_auc_frame['y'].shape)
        # print(test_auc_frame['x'].shape)

    return test_auc_frame, remove_list

def lstm_train(traindict):
    """
    All this is doing is training the lstm network.
    After training make plots to see results. (Make a plotter class or functions)
    """
    train_x,train_y = norm_train_max_min(   data = traindict,
                                            # max1=hyparams['max'],
                                            # min1=hyparams['min']
                                            max1 = max1,
                                            min1 = min1
                                        )
    
    train, val = {}, {}
    train['x'], val['x'],train['y'],val['y'] = train_test_split(    train_x,
                                                                    train_y,
                                                                    test_size = hyparams['networks']['lstm']['val_ratio']
                                                                    )
    
    # print(train['x'].shape)
    # quit()
    # took out to make tensorify func universal
    # used to do train test split in tensorify function
    train_data,val_data = tensorify(    train, 
                                        val,
                                        batch_size = hyparams['batch_size']
                                        )

    #naming convection
    nc = [  loc['nc']['date'],
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['frames'],
            ] # Note that frames is the sequence input

    # folders not saved by dates
    make_dir(loc['model_path_list']) # Make directory to save model, no deped

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
    loss_plot(history, plot_loc, nc, save_wandb=False)

    return model

def frame_traj_model_auc(model, testdict):
    """
    This function is meant to find the frame level based AUC
    model: any trajactory prediction model (would need to check input matches)
    """

    # Note that this return ious as a prob 
    test_auc_frame, remove_list = ped_auc_to_frame_auc_data(model, testdict)
    
    if test_auc_frame == 'not possible':
        quit()
    
    # 1 means  abnormal, if normal than iou would be high
    wandb_name = ['rocs', 'roc_curve']
    y_true = test_auc_frame['y']
    y_pred = test_auc_frame['x'][:,0]

    make_dir(loc['metrics_path_list'])
    plot_loc = join(    os.path.dirname(os.getcwd()),
                        *loc['metrics_path_list']
                        )    
    nc = [  loc['nc']['date'] + '_per_frame',
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['frames']
            ] # Note that frames is the sequence input

    wandb_name = ['rocs', 'roc_curve']
    roc_plot(y_true,y_pred, plot_loc, nc,wandb_name)
    print(remove_list.shape)
    

    #### Per bounding box
    nc_per_human = nc.copy()
    nc_per_human[0] = loc['nc']['date'] + '_per_bounding_box'
    y_pred_per_human = iou_as_probability(testdict, model)

    y_true_per_human = testdict['abnormal']
    #####################################################################
    # Might have a problem here in wandb if tried running and saving 
    roc_plot(y_true_per_human, y_pred_per_human, plot_loc, nc_per_human, wandb_name)


def helper_TP_TN_FP_TN(datadict, traj_model, ped):

    """
    datadict: 
    traj_model: lstm, etc
    ped: dict with x is two columns contains predictions, indices
         y contains the ground truth information 
    """
  

    print('removed ped length x: {}'.format( len( removed_ped['x'] ) ) )
    print('removed ped length y: {}'.format( len( removed_ped['y'] ) ) )
    # seperates them into TP. TN, FP, FN
    y_pred = bm_model.predict(removed_ped['x'][:,0])

    conf_dict = seperate_misclassifed_examples( y_pred = y_pred,
                                                indices = removed_ped['x'][:,1],
                                                test_y = removed_ped['y'],
                                                threshold=0.5
                                                )

    # what am I actually returning
    TP_TN_FP_FN, boxes_dict = sort_TP_TN_FP_FN_by_vid_n_frame(datadict, conf_dict )


    # Does not return result, but saves images to folders
    make_dir(loc['visual_trajectory_list'])
    pic_loc = join(     os.path.dirname(os.getcwd()),
                        *loc['visual_trajectory_list']
                        )

    # need to make last one robust "test_vid" : "train_vid"
    # can change

    loc_videos = loc['data_load'][exp['data']]['test_vid']
    # print(boxes_dict.keys())
    # quit()
    for conf_key in boxes_dict.keys():
        temp = loc['visual_trajectory_list'].copy()
        temp.append(conf_key)
        make_dir(temp)

    for conf_key in boxes_dict.keys():
        pic_loc_conf_key =  join(pic_loc, conf_key)
        cycle_through_videos(traj_model, boxes_dict[conf_key], max1, min1, pic_loc_conf_key, loc_videos, xywh=True)



def classifer_train(traindict, testdict, lstm_model):

    # Loads based on experiment 
    train, val, test = data_binary(traindict, testdict, lstm_model, max1, min1)

    print("pos:{}, neg:{}".format(pos,neg))
    neg, pos = np.bincount(train['y'])
    print(pos/neg)
    # quit()

    
    # Removing the index so I can pass into tensofify and not have two
    # functions that do similar things.
    train_no_index, val_no_index = {},{}
    val_no_index['x'] = val['x'][:,0]
    val_no_index['y'] = val['y']
 
    train_no_index['x'] = train['x'][:,0]
    train_no_index['y'] = train['y']


    train_tensor, val_tensor = tensorify(
                                            train_no_index,    # print("pos:{}, neg:{}".format(pos,neg))

                                            val_no_index,
                                            batch_size = hyparams['networks']['binary_classifier']['batch_size'])

    #naming convection
    nc = [  loc['nc']['date'],
            loc['nc']['model_name_binary_classifer'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['frames']
            ] # Note that frames is the sequence input


    make_dir(loc['model_path_list']) # Make directory to save model

    
    model_loc = join(   os.path.dirname(os.getcwd()),
                        *loc['model_path_list']
                        ) # create save link

    if not hyparams['networks']['binary_classifier']['wandb']:
        run = 'filler string'
    else:
        run = wandb.init(project="abnormal_pedestrain")    

    history, model= binary_network( train_tensor,
                                    val_tensor,
                                    model_loc=model_loc,
                                    nc =nc,
                                    weighted_binary=True, # should make clearer
                                    weight_ratio = pos/neg,
                                    output_bias =0, # Filler Play around with
                                    run = run, # this is for wandb
                                    epochs=hyparams['epochs'],
                                    save_model=hyparams['networks']['binary_classifier']['save_model'],
                                    )

    # test_auc_frame has iou on one column and index values on other column
    # remove_list contain pedestrain that are in the same frame as other
    # people in the test set for classifer

    
    # folders not saved by dates
    ###################################################################
    #  Training results
    ###################################################################
    make_dir(loc['metrics_path_list'])
    plot_loc = join(    os.path.dirname(os.getcwd()),
                        *loc['metrics_path_list']
                        )

    loss_plot(history, plot_loc, nc, save_wandb = True)
    accuracy_plot(history, plot_loc,nc)


    ###################################################################
    #  Testing results
    ###################################################################
    
    # Removes the index 
    test_no_index = {}
    test_no_index['x'] = test['x'][:,0]
    test_no_index['y'] = test['y']
    wandb_name = ['rocs', 'roc_curve']
    
    y_pred = model.predict(test_no_index['x'])
    y_true = test_no_index['y']
    roc_plot(y_true, y_pred, plot_loc, nc,wandb_name)
    
    # if exp['3_1']:
    test_auc_frame, removed_ped_index = ped_auc_to_frame_auc_data(model, testdict, test)
    # elif exp['3_2']:
    # For 3_2
    ########################################################################################################
    # Need to seperate test based on if their are negative index values
    # Note that negative index values means that selected test frame
    # came from the orgininal training frame in dictornary
        # from_traindict_index = np.where(test['x'][:,1] < 0)[0] # cuz of initializing the 0 of train as -0.01
        # from_testdict_index = np.where(test['x'][:,1] >= 0)[0] 
        
        # from_traindict  = {}
        # from_traindict['x'] = test['x'][from_traindict_index]
        # from_traindict['y'] = test['y'][from_traindict_index]
        # from_traindict_auc_frame, from_traindict_removed_ped_index = ped_auc_to_frame_auc_data(model, traindict, from_traindict)
        
        # from_testdict = {}
        # from_testdict['x'] = test['x'][from_testdict_index]
        # from_testdict['y'] = test['y'][from_testdict_index]
        # from_testdict_auc_frame, from_testdict_removed_ped_index = ped_auc_to_frame_auc_data(model, testdict, from_testdict)

        # test_auc_frame = {}
        # test_auc_frame['x'] = np.append(from_traindict_auc_frame['x'], from_testdict_auc_frame['x'], axis = 0)
        # test_auc_frame['y'] = np.append(from_traindict_auc_frame['y'], from_testdict_auc_frame['y'])
    ########################################################################################################



    # Removes the index 
    nc_frame = nc.copy()
    nc_frame[0] = loc['nc']['date'] + 'frame'
    test_no_index = {}
    test_no_index['x'] = test_auc_frame['x'][:,0]
    test_no_index['y'] = test_auc_frame['y']
    wandb_name_frame = ['rocs_frame', 'roc_curve_frame']
    print("this is nc_frame:{}".format(nc_frame))
    print("this is nc:{}".format(nc))

    y_pred = model.predict(test_no_index['x'])
    y_true = test_no_index['y']

    roc_plot(y_true, y_pred, plot_loc, nc_frame, wandb_name_frame)

    # should allow me to debug safety without abort program
    # run.finish()

    # quit()
    #########################################
    # More of a plotter than anything
    # Make into a function 
    # Looking at pedestrains that are deleted
    if len(removed_ped_index) >= 1:
        removed_ped = {}
        removed_ped['x'] = test['x'][removed_ped_index, :]
        removed_ped['y'] = test['y'][removed_ped_index]

        print('removed ped length x: {}'.format( len( removed_ped['x'] ) ) )
        print('removed ped length y: {}'.format( len( removed_ped['y'] ) ) )
        # seperates them into TP. TN, FP, FN
        conf_dict = seperate_misclassifed_examples( bm_model = model,
                                                    test_x = removed_ped['x'][:,0],
                                                    indices = removed_ped['x'][:,1],
                                                    test_y = removed_ped['y'],
                                                    threshold=0.5
                                                    )

        # print(len(conf_dict['TN']))
        # print(len(conf_dict['FN']))
        # print(len(conf_dict['FP']))
        # print(len(conf_dict['TP']))
        # print(conf_dict['TN'])
        # print(conf_dict['FN'])
        # print(conf_dict['FP'])
        # print(conf_dict['TP'])

        # quit()
        # what am I actually returning
        TP_TN_FP_FN, boxes_dict = sort_TP_TN_FP_FN_by_vid_n_frame(testdict, conf_dict )


        # Does not return result, but saves images to folders
        make_dir(loc['visual_trajectory_list'])
        pic_loc = join(     os.path.dirname(os.getcwd()),
                            *loc['visual_trajectory_list']
                            )

        # need to make last one robust "test_vid" : "train_vid"
        # can change

        loc_videos = loc['data_load'][exp['data']]['test_vid']
        # print(boxes_dict.keys())
        # quit()
        for conf_key in boxes_dict.keys():
            temp = loc['visual_trajectory_list'].copy()
            temp.append(conf_key)
            make_dir(temp)

        for conf_key in boxes_dict.keys():
            pic_loc_conf_key =  join(pic_loc, conf_key)
            cycle_through_videos(lstm_model, boxes_dict[conf_key], max1, min1, pic_loc_conf_key, loc_videos, xywh=True)

    print("pos:{}, neg:{}".format(pos,neg))

    # go back and fix filler
    print('go back and fix filler')
    
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
    # To-Do add input argument for when loading 
    load_lstm_model = True
    special_load = True # go back and clean up with command line inputs
    model_loc = join(   os.path.dirname(os.getcwd()),
                        *loc['model_path_list']
                        ) # create save link
    # 01_07_2021_lstm_network_xywh_hr-st_20
    
    nc = [  #loc['nc']['date'],
            '12_18_2020',
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['frames'],
            ] # Note that frames is the sequence input

    traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                        loc['data_load'][exp['data']]['test_file']
                                        )

    # This is a temp solution, permant is to make function normalize function
    global max1, min1
    max1 = traindict['x_ppl_box'].max()
    min1 = traindict['x_ppl_box'].min()
    


    if load_lstm_model:        
        if special_load:
            # model_path = os.path.join( os.path.dirname(os.getcwd()),
            #                             'results_all_datasets/experiment_3_1/saved_model/01_07_2021_lstm_network_xywh_avenue_20.h5'
            #                             )

            model_path = os.path.join( os.path.dirname(os.getcwd()),
                                        'results_all_datasets/experiment_3_1/saved_model/12_18_2020_lstm_network_xywh_st_20.h5'
                                        )
        else:
            model_path = os.path.join(  model_loc,
                            '{}_{}_{}_{}_{}.h5'.format(*nc)
                            )
        print(model_path)
        lstm_model = tf.keras.models.load_model(    model_path,  
                                                    custom_objects = {'loss':'mse'} , 
                                                    compile=True
                                                    )
    else:
        # returning model right now but might change that in future and load instead
        lstm_model = lstm_train(traindict)


    # classifer_train(traindict, testdict, lstm_model)
    frame_traj_model_auc(lstm_model, testdict)



    ## To_DO:
    """"
    1)  Fix the wandb saved metrics plots,
        make config file more consistent with how wandb should be call
    Why do I need an initial bias for last layer. I think I got idea
    from google. But is it advantagous.
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

    2)  Looks like I can save the hyprmas file as well in a txt file.
        Might be useful might not be



    4)  create a testing lstm. This would mean I need to return the model
        of lstm_train. Or more robustly just read saved model instead.
        Problem is if I'm running a test where don't want to save anything
        how do I do that. Maybe move them to tmp

    """



if __name__ == '__main__':
    print('GPU is on: {}'.format(gpu_check() ) )

    main()

    print('Done') 