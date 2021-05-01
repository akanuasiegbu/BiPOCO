import cv2
import numpy as np
import pandas as pd
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
from load_data import norm_train_max_min, load_pkl
from load_data_binary import compute_iou
# Plots
from metrics_plot import *

# Models
from models import lstm_network, binary_network
import wandb
from custom_functions.ped_sequence_plot import ind_seq_dict, plot_sequence, plot_frame

from custom_functions.convert_frames_to_videos import convert_spec_frames_to_vid

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
    if model =='bitrap':
        y = testdict['y_ppl_box'] # this is the gt 
        gt_bb_unorm_tlbr = xywh_tlbr(np.squeeze(y))
        predicted_bb_unorm_tlbr = xywh_tlbr(testdict['pred_trajs'])
        iou = bb_intersection_over_union_np(    predicted_bb_unorm_tlbr,
                                                gt_bb_unorm_tlbr )
        # need to squeeze to index correctly 
        iou = np.squeeze(iou)

    else:
        x,y = norm_train_max_min(   testdict,
                                    # max1 = hyparams['max'],
                                    # min1 = hyparams['min']
                                    max1 = max1,
                                    min1 = min1
                                    )

        iou = compute_iou(x, y, max1, min1,  model)

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
    could make more computailly efficent by
    returning iou_prob in this function for prob per human of
    being abnormal

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

        #############################
        # comment out after listening

       

        #############################
        
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

            else:
                same_vid_frame = np.where(unique_inverse == i )[0]
                y_pred = model.predict(test_bin['x'][:,0][same_vid_frame])
                # find max y_pred input other indices to remove list

            # This saves it for both cases below  
            max_loc = np.where( y_pred == np.max(y_pred))[0]
            if len(max_loc) > 1:
                temp_1 = max_loc[1:]
                temp_2 = np.where(y_pred != np.max(y_pred))[0]
                remove_list_temp.append(same_vid_frame[temp_1])
                remove_list_temp.append(same_vid_frame[temp_2])
            else:
                temp = np.where(y_pred != np.max(y_pred))[0]
                remove_list_temp.append(same_vid_frame[temp])
                    
        
        remove_list = [item for sub_list in remove_list_temp for item in sub_list]

             
        remove_list = np.array(remove_list).astype(int)

        # print('Length of removed elements is :{}'.format(len(remove_list)))
        # print(test_bin['x'].shape)
        test_auc_frame = {}
        if not test_bin:
            iou_prob_per_person = np.append(iou_prob.reshape(-1,1), test_index.reshape(-1,1), axis=1)
            test_auc_frame['x'] = np.delete( iou_prob_per_person, remove_list, axis = 0 )
            test_auc_frame['y'] = np.delete(testdict['abnormal'].reshape(-1,1) , remove_list, axis=0)
        else:
            test_auc_frame['x'] = np.delete(test_bin['x'], remove_list, axis=0)
            test_auc_frame['y'] = np.delete(test_bin['y'], remove_list, axis=0)
        # print(test_auc_frame['y'].shape)
        # print(test_auc_frame['x'].shape)

        y_pred_per_human = iou_prob

    return test_auc_frame, remove_list, y_pred_per_human


def frame_traj_model_auc(model, testdict):
    """
    This function is meant to find the frame level based AUC
    model: any trajactory prediction model (would need to check input matches)
    """

    # Note that this return ious as a prob 
    #test_auc_frame, remove_list = ped_auc_to_frame_auc_data(model, testdict)
    test_auc_frame, remove_list, y_pred_per_human = ped_auc_to_frame_auc_data('bitrap', testdict)
    
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


    # uncomment to plot video frames
    # print("Number of abnormal people after maxed {}".format(sum(test_auc_frame['y'])))
    print("Number of abnormal people after maxed {}".format(len(np.where(test_auc_frame['y'] == 1 )[0] ) ))

    # helper_TP_TN_FP_FN( datadict = testdict, 
    #                     traj_model = model, 
    #                     ped = test_auc_frame, 
    #                     both=True
    #                     )

    # FOR PLOTTING ALL THE DATA
    test_auc_frame_all = {}
    iou_prob_per_person = np.append(y_pred_per_human.reshape(-1,1), np.arange(0,len(y_pred_per_human)).reshape(-1,1), axis=1)
    test_auc_frame_all['x'] = iou_prob_per_person
    test_auc_frame_all['y'] = testdict['abnormal_ped'].reshape(-1,1) 


    helper_TP_TN_FP_FN( datadict = testdict, 
                        traj_model = model, 
                        ped = test_auc_frame_all, 
                        both=True
                        )

    # this is for plotting indivual people make into a func
   
    


    #### Per bounding box
    nc_per_human = nc.copy()
    nc_per_human[0] = loc['nc']['date'] + '_per_bounding_box'
    # y_pred_per_human = iou_as_probability(testdict, model)

    abnormal_index = np.where(testdict['abnormal_ped'] == 1)
    normal_index = np.where(testdict['abnormal_ped'] == 0)

    # Uncomment to make iou plots
    ################################################

    plot_iou(   prob_iou = y_pred_per_human[abnormal_index[0]],
                xlabel ='Detected Abnormal Pedestrains ',
                ped_type = 'abnormal_ped',
                plot_loc = plot_loc,
                nc = nc_per_human
                )

    plot_iou(   prob_iou = y_pred_per_human[normal_index[0]],
                xlabel ='Detected Normal Pedestrains ',
                ped_type = 'normal_ped',
                plot_loc = plot_loc,
                nc = nc_per_human
                )


    abnormal_index_frame = np.where(test_auc_frame['y'] == 1)
    normal_index_frame = np.where(test_auc_frame['y'] == 0)




    plot_iou(   prob_iou = test_auc_frame['x'][abnormal_index_frame[0], 0],
                xlabel ='Detected Abnormal Pedestrains ',
                ped_type = 'abnormal_ped_frame',
                plot_loc = plot_loc,
                nc = nc_per_human
                )

    plot_iou(   prob_iou = test_auc_frame['x'][normal_index_frame[0], 0],
                xlabel ='Detected Normal Pedestrains ',
                ped_type = 'normal_ped_frame',
                plot_loc = plot_loc,
                nc = nc_per_human
                )


    
    ###################################################

    y_true_per_human = testdict['abnormal_ped']
    #####################################################################
    # Might have a problem here in wandb if tried running and saving 
    roc_plot(y_true_per_human, y_pred_per_human, plot_loc, nc_per_human, wandb_name)


def helper_TP_TN_FP_FN(datadict, traj_model, ped, both):

    """
    This uses function in the TP_TN_FP_FN file for plotting
    datadict: 
    traj_model: lstm, etc
    ped: dict with x is two columns contains predictions, indices
         y contains the ground truth information 
    both: plot bitrap and lstm model on top of each other
    """
  

    # seperates them into TP. TN, FP, FN

    # Note that y_pred should not be threshold yet, granted if it is no
    # error cuz would change by threshold again assuming using same threshold 
    conf_dict = seperate_misclassifed_examples( y_pred = ped['x'][:,0],
                                                indices = ped['x'][:,1],
                                                test_y = ped['y'],
                                                threshold=0.5
                                                )

    
    print('length of  TP {} '.format(len(conf_dict['TP'])))
    print('length of  TN {} '.format(len(conf_dict['TN'])))
    print('length of  FP {} '.format(len(conf_dict['FP'])))
    print('length of  FN {} '.format(len(conf_dict['FN'])))
    # quit()
    
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
        cycle_through_videos(traj_model, both, boxes_dict[conf_key], max1, min1, pic_loc_conf_key, loc_videos, xywh=True)



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


# def trouble_shot(testdict, model, frame, ped_id, vid):
def plot_traj_gen_traj_vid(testdict, model):
    """
    testdict: the dict
    model: model to look at
    frame: frame number of interest (note assuming that we see the final frame first and)
            then go back and plot traj (int)
    ped_id: pedestrain id (int)
    vid: video number (string)

    """

    # This is helping me plot the data from tlbr -> xywh -> tlbr
    ped_loc = loc['visual_trajectory_list'].copy()
    frame = 517
    ped_id = 11
    
    # vid = '07_0009'
    vid = '02'
    # loc_videos = loc['data_load'][exp['data']]['test_vid']
    

    # Video to select ,not using config file to give freedom to select

    # loc_videos = "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/st/test_vid/{}_st_output_test.avi".format(vid)
    # loc_videos = "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/avenue/test_vid/{}_output_yolo_test.avi".format(vid)
    loc_videos = "/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/testing_videos/{}.avi".format(vid)
    # loc_videos = '/mnt/roahm/users/akanu/projects/Deep-SORT-YOLOv4/tensorflow2.0/deep-sort-yolov4/input_video/st_test/{}.avi'.format(vid)



    
    
    ped_loc[-1] =  '{}'.format(vid) + '_' + '{}'.format(frame)+ '_' + '{}'.format(ped_id)
    make_dir(ped_loc)
    pic_loc = join(     os.path.dirname(os.getcwd()),
                        *ped_loc
                        )


    # test_auc_frame, remove_list, y_pred_per_human = ped_auc_to_frame_auc_data(model, testdict)
    
    # temp_dict = {}
    # for i in testdict.keys():
    #     indices = np.array(test_auc_frame['x'][:,1], dtype=int)
    #     temp_dict[i] = testdict[i][indices]



    person_seq = ind_seq_dict(testdict, '{}'.format(vid), frame,  ped_id) # this is a slow search I would think
    # person_seq = ind_seq_dict(temp_dict, '{}'.format(vid), frame,  ped_id) # this is a slow search I would think
    
    # test_auc_frame, remove_list, y_pred_per_human = ped_auc_to_frame_auc_data(model, testdict)

    # print('in xywh coordinate')
    # print(person_seq['x_ppl_box'])
    # xx, yy = norm_train_max_min(person_seq, max1, min1, undo_norm=False)
    # xx = np.expand_dims(xx, axis=0)
    # yy = np.expand_dims(yy, axis=0)

    # bbox_pred_norm = model.predict(xx)
    # print('bbox_pred_norm {}'.format(xywh_tlbr(bbox_pred_norm)))
    # print('bb gt {}'.format(xywh_tlbr(yy)))


    # I still want to see what it looks like in the normaized coordinates to know
    # if it should change
    # iou_norm = bb_intersection_over_union_np(   xywh_tlbr(bbox_pred_norm),
    #                                             xywh_tlbr(yy)
    #                                         )
    # print('iou in normilized coordinate {}'.format(iou_norm))
    
  

    

    # bbox_pred = norm_train_max_min(bbox_pred_norm, max1, min1, undo_norm=True)
    bbox_pred = np.expand_dims(person_seq['pred_trajs'], axis=0)
    iou_unorm = bb_intersection_over_union_np(  xywh_tlbr(bbox_pred),
                                                xywh_tlbr(np.expand_dims(person_seq['y_ppl_box'], axis=0) )
                                                )
                                        
    # print('bbox_pred in standard coordinates predicted {}'.format(xywh_tlbr(bbox_pred)))
    # print('bbox gt {}'.format(xywh_tlbr(np.expand_dims(person_seq['y_ppl_box'], axis=0))) )
    print('iou not normalized  which is correct{}'.format(iou_unorm))
    
    # # see vid frame i
    print('vid:{} frame:{} id:{}'.format(vid, frame, ped_id))
    print('abnormal indictor {}'.format(person_seq['abnormal_ped']))

    # quit()
    plot_sequence(  person_seq,
                    max1,
                    min1,
                    '{}.txt'.format(vid),
                    pic_loc = pic_loc,
                    loc_videos = loc_videos,
                    xywh= True
                    )

    gen_vid(vid_name = '{}_{}_{}'.format(vid, frame, ped_id),pic_loc = pic_loc, frame_rate = 1)
    print('should see this rn if quit works')
    quit()


   

def gen_vid(vid_name, pic_loc, frame_rate):
    # vid_name = '04_670_61'
    # image_loc = '/home/akanu/results_all_datasets/experiment_traj_model/visual_trajectory_consecutive/{}'.format(vid_name)
    save_vid_loc = loc['visual_trajectory_list']
    save_vid_loc[-1] = 'short_generated_videos'

    make_dir(save_vid_loc)
    save_vid_loc = join(     os.path.dirname(os.getcwd()),
                            *save_vid_loc
                            )
    convert_spec_frames_to_vid( loc = pic_loc, 
                                save_vid_loc = save_vid_loc, 
                                vid_name = vid_name, frame_rate = frame_rate )

    

def main():
    

    traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                        loc['data_load'][exp['data']]['test_file']
                                        )

    quit()

    # To-Do add input argument for when loading 
    load_lstm_model = True
    special_load = False # go back and clean up with command line inputs
    model_loc = join(   os.path.dirname(os.getcwd()),
                        *loc['model_path_list']
                        ) # create save link
    
    
    nc = [  #loc['nc']['date'],
            '03_11_2021',
            # loc['nc']['model_name'],
            'lstm_network',
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['frames'],
            ] # Note that frames is the sequence input



  
    # This is a temp solution, perm is to make function normalize function
    
    # trouble_shot(testdict)

    #  Note I don't need a model to do trobule shot code


    if load_lstm_model:        
        if special_load:
            # model_path = os.path.join( os.path.dirname(os.getcwd()),
            #                             'results_all_datasets/experiment_3_1/saved_model/01_07_2021_lstm_network_xywh_avenue_20.h5'
            #                             )

            # model_path = os.path.join( os.path.dirname(os.getcwd()),
            #                             'results_all_datasets/experiment_3_1/saved_model/12_18_2020_lstm_network_xywh_st_20.h5'
            #                             )
            pass
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

    #Load Data
    pkldict = load_pkl()
    traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                        loc['data_load'][exp['data']]['test_file']
                                        )
    global max1, min1
    max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
    min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()
    
    # frame_traj_model_auc(lstm_model, pkldict)
     # Note would need to change mode inside frame_traj


    # classifer_train(traindict, testdict, lstm_model)
    # frame_traj_model_auc(lstm_model, testdict)
     

    plot_traj_gen_traj_vid(pkldict,lstm_model)

    



if __name__ == '__main__':
    # print('GPU is on: {}'.format(gpu_check() ) )


    main()

    print('Done') 