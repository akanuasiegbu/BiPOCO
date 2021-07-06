import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow import keras
import os, sys, time
from os.path import join
from custom_functions.utils import make_dir, SaveTextFile, write_to_txt, SaveAucTxt, SaveAucTxtTogether

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

# from custom_functions.convert_frames_to_videos import convert_spec_frames_to_vid

from custom_functions.auc_metrics import l2_error, iou_as_probability, anomaly_metric, combine_time
from custom_functions.auc_metrics import giou_as_metric, ciou_as_metric, diou_as_metric

from verify import order_abnormal

from custom_functions.visualizations import plot_frame_from_image, plot_vid


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
            hyparams['input_seq'],
            hyparams['pred_seq']
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

def ped_auc_to_frame_auc_data(model, testdicts, metric, avg_or_max, test_bin=None):
    """
    Note that this does not explictly calcuate frame auc
    but removes select points to reduce to frame AUC data.

    testdict: From orginal data test dict
    test_bin: binary classifer test dict
    model: binary classifer model or lstm model


    return:
    test_auc_frame: people that are in the frame as proability and index of frame
    remove_list: indices of pedestrain removed
    """
    if not test_bin:
        # calc iou prob
        # Note for iou, giou, ciou and diou. used i-iou, 1-giou, 1-ciou, 1-diou etc 
        # because abnormal pedestrain would have a higher score
        if metric == 'iou':
            prob = iou_as_probability(testdicts, model, errortype = hyparams['errortype'], max1 = max1, min1= min1)
        
        elif metric == 'l2':
            
            prob = l2_error(testdicts = testdicts, models = model, errortype =hyparams['errortype'], max1=max1,min1= min1)
        
        elif metric == 'giou':
            prob = giou_as_metric(testdicts = testdicts, models = model, errortype =hyparams['errortype'], max1=max1,min1= min1)

        elif metric == 'ciou':
            prob = ciou_as_metric(testdicts = testdicts, models = model, errortype =hyparams['errortype'], max1=max1,min1= min1)
        
        elif metric == 'diou':
            prob = diou_as_metric(testdicts = testdicts, models = model, errortype =hyparams['errortype'], max1=max1,min1= min1)

        pkldicts = combine_time(    testdicts, models=model, errortype=hyparams['errortype'], 
                                    modeltype = exp['model_name'], max1 =max1, min1=min1)


        out = anomaly_metric(   prob, 
                                avg_or_max,
                                pred_trajs = pkldicts['pred_trajs'],
                                gt_box = pkldicts['gt_bbox'], 
                                vid_loc = pkldicts['vid_loc'],
                                frame_loc = pkldicts['frame_y'],
                                person_id = pkldicts['id_y'],
                                abnormal_gt = pkldicts['abnormal_gt_frame'],
                                abnormal_person = pkldicts['abnormal_ped_pred'])

        prob = out['prob']
        vid_loc = out['vid']
        frame_loc = out['frame']
        
        # frame_loc, vid_loc, abnormal_gt_frame_metric, std, std_iou
        test_index = np.arange(0, len(out['abnormal_gt_frame_metric']), 1)
    
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
    
    repeat_inverse_id = np.where(unique_counts>1)[0] # this works because removing those greater than 1

    
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
                y_pred = prob[same_vid_frame]

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
            test_auc_frame['x'] = np.delete( prob.reshape(-1,1), remove_list, axis = 0 )
            test_auc_frame['y'] = np.delete(out['abnormal_gt_frame_metric'], remove_list, axis = 0)
            test_auc_frame['x_pred_per_human'] = prob.reshape(-1,1)
            test_auc_frame['y_pred_per_human'] = out['abnormal_gt_frame_metric']
            test_auc_frame['std_per_frame'] = np.delete(out['std'], remove_list, axis = 0)
            test_auc_frame['std_per_human'] = out['std']
            test_auc_frame['index'] = test_index.reshape(-1,1)

            # if hyparams['metric'] == 'iou':
            test_auc_frame['std_iou_or_l2_per_frame'] = np.delete(out['std_iou_or_l2'], remove_list, axis = 0)
            test_auc_frame['std_iou_or_l2_per_human'] = out['std_iou_or_l2']

            out_frame = {}
            for key in out.keys():
                out_frame[key] = np.delete(out[key], remove_list, axis = 0)
            # test_auc_frame['y'] = np.delete(testdict['abnormal_gt_frame'].reshape(-1,1) , remove_list, axis=0)
            # test_auc_frame['abnormal_ped_pred'] = np.delete(testdict['abnormal_ped_pred'].reshape(-1,1) , remove_list, axis=0)
        else:
            test_auc_frame['x'] = np.delete(test_bin['x'], remove_list, axis=0)
            test_auc_frame['y'] = np.delete(test_bin['y'], remove_list, axis=0)
        # print(test_auc_frame['y'].shape)
        # print(test_auc_frame['x'].shape)

    return test_auc_frame, remove_list, out_frame


def frame_traj_model_auc(model, testdicts, metric, avg_or_max):
    """
    This function is meant to find the frame level based AUC
    model: any trajactory prediction model (would need to check input matches)
    testdicts: is the input data dict
    metric: iou or l2 metric
    avg_or_max: used when looking at same person over frame, for vertical comparssion
    """

    # Note that this return ious as a prob 
    test_auc_frame, remove_list, out_frame = ped_auc_to_frame_auc_data(model, testdicts, metric, avg_or_max)
    # test_auc_frame, remove_list, y_pred_per_human = ped_auc_to_frame_auc_data('bitrap', testdict)
    
    
    if test_auc_frame == 'not possible':
        quit()
    
    # 1 means  abnormal, if normal than iou would be high
    wandb_name = ['rocs', 'roc_curve']
    
    path_list = loc['metrics_path_list'].copy()
    visual_path = loc['visual_trajectory_list'].copy()
    for path in [path_list, visual_path]:
        path.append('{}_{}_in_{}_out_{}_K_{}'.format(loc['nc']['date'], exp['data'], hyparams['input_seq'],
                                                            hyparams['pred_seq'],exp['K'] ))
        path.append('{}_{}_{}_in_{}_out_{}_{}'.format(  loc['nc']['date'],
                                                        metric,
                                                        avg_or_max, 
                                                        hyparams['input_seq'], 
                                                        hyparams['pred_seq'],
                                                        hyparams['errortype'] ) )
    make_dir(path_list)
    plot_loc = join( os.path.dirname(os.getcwd()), *path_list )
    joint_txt_file_loc = join( os.path.dirname(os.getcwd()), *path_list[:-1] )

    print(joint_txt_file_loc)
    # quit()
    # This saves the average metrics into a text file
    ouput_with_metric, auc_frame_human = write_to_txt(test_auc_frame)
    file_avg_metrics = SaveTextFile(plot_loc, metric)
    file_avg_metrics.save(ouput_with_metric, auc_frame_human)
    file_with_auc = SaveAucTxt(joint_txt_file_loc, metric)
    file_with_auc.save(auc_frame_human)
    # quit()

    # # For visualzing 
    # # This makes folders for the videos
    # for i in range(1,22):
    #     path = visual_path.copy()
    #     path.append('{:02d}'.format(i))
    #     make_dir(path)

    # visual_plot_loc = join( os.path.dirname(os.getcwd()), *visual_path )

    nc = [  loc['nc']['date'] + '_per_frame',
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['input_seq'],
            hyparams['pred_seq']
            ] # Note that frames is the sequence input
    


    # # This plots the data for visualizations
    # pic_locs = loc['data_load']['avenue']['pic_loc_test']
    # plot_vid( out_frame, pic_locs, visual_plot_loc )

    # quit()

    # uncomment to plot video frames
    # print("Number of abnormal people after maxed {}".format(sum(test_auc_frame['y'])))
    print("Number of abnormal people after maxed {}".format(len(np.where(test_auc_frame['y'] == 1 )[0] ) ))

   


    #### Per bounding box
    nc_per_human = nc.copy()
    nc_per_human[0] = loc['nc']['date'] + '_per_bounding_box'
    # y_pred_per_human = iou_as_probability(testdict, model)

    # abnormal_index = np.where(testdict['abnormal_ped_pred'] == 1)
    # normal_index = np.where(testdict['abnormal_ped_pred'] == 0)
    
    abnormal_index = np.where(test_auc_frame['y_pred_per_human'] == 1)[0]
    normal_index = np.where(test_auc_frame['y_pred_per_human'] == 0)[0]
    
    abnormal_index_frame = np.where(test_auc_frame['y'] == 1)[0]
    normal_index_frame = np.where(test_auc_frame['y'] == 0)[0]
    
    if metric == 'iou':
        ylabel = '1-IOU'

    elif metric == 'l2':
        ylabel = 'L2 Error'

    elif metric =='giou':
        ylabel ='giou'

    elif metric =='ciou':
        ylabel = 'ciou'

    elif metric =='diou':
        ylabel = 'diou'

    index = [abnormal_index, normal_index]
    ped_type = ['abnormal_ped', 'normal_ped']
    xlabel = ['Detected Abnormal Pedestrains', 'Detected Normal Pedestrains']
    titles =['Abnormal', 'Normal']


    ##############
    # # DELETE OR MOVE TO A Different place
    index = [abnormal_index_frame, normal_index_frame ]
    xlabel = ['Abnormal Frames', 'Detected Normal Frames']
    ped_type = ['abnormal_ped_frame', 'normal_ped_frame']
    wandb_name = ['rocs', 'roc_curve']
    
    y_true = test_auc_frame['y']
    y_pred = test_auc_frame['x']

    # Uncomment to make iou plots
    ################################################

    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        plot_iou(   prob_iou = test_auc_frame['x_pred_per_human'][indices],
                    gt_label = test_auc_frame['y_pred_per_human'][indices],
                    xlabel = x_lab,
                    ped_type = ped_,
                    plot_loc = plot_loc,
                    nc = nc_per_human,
                    ylabel = ylabel,
                    title = title
                    )        
    # xlabel = ['Detected Abnormal Pedestrains', 'Detected Normal Pedestrains']
    index = [abnormal_index_frame, normal_index_frame ]
    xlabel = ['Abnormal Frames', 'Detected Normal Frames']
    ped_type = ['abnormal_ped_frame', 'normal_ped_frame']

    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        plot_iou(   prob_iou = np.sum(test_auc_frame['std_per_frame'][indices], axis = 1),
                    gt_label = test_auc_frame['y'][indices],
                    xlabel = x_lab,
                    ped_type = '{}_std'.format(ped_),
                    plot_loc = plot_loc,
                    nc = nc,
                    ylabel = 'Standard Deviation Summed',
                    title = title
                    )

    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        for i, axis in zip(range(0,4), ['Mid X', 'Mid Y', 'W', 'H']):
            plot_iou(   prob_iou = test_auc_frame['std_per_frame'][indices][:,i],
                        gt_label = test_auc_frame['y'][indices],
                        xlabel = x_lab,
                        ped_type = '{}_std_axis_{}'.format(ped_, i),
                        plot_loc = plot_loc,
                        nc = nc,
                        ylabel = 'Standard Deviation {}'.format(axis),
                        title = '{}_axis_{}'.format(title, i)
                        )

    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        plot_iou(   prob_iou = test_auc_frame['std_iou_or_l2_per_frame'][indices],
                    gt_label = test_auc_frame['y'][indices],
                    xlabel = x_lab,
                    ped_type = '{}_std_{}'.format(ped_, hyparams['metric']),
                    plot_loc = plot_loc,
                    nc = nc,
                    ylabel = 'Standard Deviation {}'.format(hyparams['metric']),
                    title = title 
                    )


    for indices, ped_, x_lab, title in zip(index, ped_type, xlabel, titles ):
        plot_iou(   prob_iou = test_auc_frame['x'][indices],
                    gt_label = test_auc_frame['y'][indices],
                    xlabel = x_lab,
                    ped_type = ped_,
                    plot_loc = plot_loc,
                    nc = nc,
                    ylabel = ylabel,
                    title = title
                    )        

                

    
    ###################################################
    # This is where the ROC Curves are plotted 
    wandb_name = ['rocs', 'roc_curve']
    
    y_true = test_auc_frame['y']
    y_pred = test_auc_frame['x']

    # y_true_per_human = testdict['abnormal_ped_pred']
    y_true_per_human = test_auc_frame['y_pred_per_human']
    y_pred_per_human = test_auc_frame['x_pred_per_human']
    # Might have a problem here in wandb if tried running and saving 
    #####################################################################
    roc_plot( y_true_per_human, y_pred_per_human, plot_loc, nc_per_human, wandb_name)
    roc_plot( y_true, y_pred, plot_loc, nc, wandb_name)

    return auc_frame_human





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

    frame = 517
    # This is helping me plot the data from tlbr -> xywh -> tlbr
    ped_loc = loc['visual_trajectory_list'].copy()
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
    print('abnormal indictor {}'.format(person_seq['abnormal_ped_pred']))

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
    save_vid_loc = join(    os.path.dirname(os.getcwd()),
                            *save_vid_loc
                            )
    convert_spec_frames_to_vid( loc = pic_loc, 
                                save_vid_loc = save_vid_loc, 
                                vid_name = vid_name, frame_rate = frame_rate )

    

def main():
    
    # To-Do add input argument for when loading 
    load_lstm_model = True
    special_load = False # go back and clean up with command line inputs
    model_loc = join(   os.path.dirname(os.getcwd()),
                        *loc['model_path_list']
                        ) # create save link
    



    print(model_loc)
    
    nc = [  #loc['nc']['date'],
            '03_11_2021',
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['input_seq'],
            hyparams['pred_seq'],
            ] # Note that frames is the sequence input


    traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                        loc['data_load'][exp['data']]['test_file'],
                                        hyparams['input_seq'], hyparams['pred_seq'] 
                                        )
    global max1, min1
    
    max1 = None
    min1 = None
    max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
    min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()

  
    # This is a temp solution, perm is to make function normalize function
    
    # trouble_shot(testdict)

    #  Note I don't need a model to do trobule shot code


    # if load_lstm_model:        
    #     if special_load:
    #         # model_path = os.path.join( os.path.dirname(os.getcwd()),
    #         #                             'results_all_datasets/experiment_3_1/saved_model/01_07_2021_lstm_network_xywh_avenue_20.h5'
    #         #                             )

    #         # model_path = os.path.join( os.path.dirname(os.getcwd()),
    #         #                             'results_all_datasets/experiment_3_1/saved_model/12_18_2020_lstm_network_xywh_st_20.h5'
    #         #                             )
    #         pass
    #     else:
    #         model_path = os.path.join(  model_loc,
    #                         '{}_{}_{}_{}_{}_{}.h5'.format(*nc)
    #                         )
    #     print(model_path)
    #     lstm_model = tf.keras.models.load_model(    model_path,  
    #                                                 custom_objects = {'loss':'mse'} , 
    #                                                 compile=True
    #                                                 )
    # else:
    #     # returning model right now but might change that in future and load instead
    #     lstm_model = lstm_train(traindict)

    #Load Data
    # pkldicts_temp = load_pkl('/home/akanu/output_bitrap/avenue_unimodal/gaussian_avenue_in_5_out_1_K_1.pkl')
    # pkldicts = [ load_pkl(loc['pkl_file'][exp['data']]) ]
    run_quick()
    # pkldicts = []
    # pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(20,5)))
    # pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(20,10)))
    # pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(5,5)))
    # pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(5,10)))
    

    # frame_traj_model_auc([lstm_model], [testdict], hyparams['metric'], hyparams['avg_or_max'])
    # frame_traj_model_auc( 'bitrap', pkldicts, hyparams['metric'], hyparams['avg_or_max'])
    print('Input Seq: {}, Output Seq: {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
    print('Metric: {}, avg_or_max: {}'.format(hyparams['metric'], hyparams['avg_or_max']))
    # # Note would need to change mode inside frame_traj


    # classifer_train(traindict, testdict, lstm_model)
     

    # plot_traj_gen_traj_vid(pkldict,lstm_model)

    
def run_quick():
    in_lens =[3,5,5,13,20,20,25]
    out_lens = [3,5,10,13,5,10,25]

    for in_len, out_len in zip(in_lens, out_lens):
        hyparams['input_seq'] = in_len
        hyparams['pred_seq'] = out_len
        print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
        # continue
        if exp['data'] == 'st':
            pklfile = loc['pkl_file']['st_template'].format(hyparams['input_seq'], hyparams['pred_seq'], exp['K'])

        elif exp['data'] =='avenue':
            pklfile = loc['pkl_file']['avenue_template'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'])

        print(pklfile)                                                                                
        pkldicts =  load_pkl(pklfile) 
        
        for error in ['error_diff', 'error_summed', 'error_flattened']:
            hyparams['errortype'] = error
            auc_metrics_list = []
            print(hyparams['errortype'])
            for metric in ['giou', 'l2', 'ciou', 'diou', 'iou']:
                hyparams['metric'] = metric
                print(hyparams['metric'])
                auc_metrics_list.append(frame_traj_model_auc( 'bitrap', [pkldicts], hyparams['metric'], hyparams['avg_or_max']))
            
            path_list = loc['metrics_path_list'].copy()
            path_list.append('{}_{}_in_{}_out_{}_K_{}'.format(loc['nc']['date'], exp['data'], hyparams['input_seq'],
                                                                hyparams['pred_seq'],exp['K'] ))
            joint_txt_file_loc = join( os.path.dirname(os.getcwd()), *path_list )

            print(joint_txt_file_loc)
            auc_together=np.array(auc_metrics_list)


            auc_slash_format = SaveAucTxtTogether(joint_txt_file_loc)
            auc_slash_format.save(auc_together)


   
   


    

if __name__ == '__main__':
    # print('GPU is on: {}'.format(gpu_check() ) )


    main()

    print('Done') 