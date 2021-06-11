# This is in develpment rn
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys, time
from os.path import join


from experiments_code.config import hyparams, loc, exp

from custom_metrics import bb_intersection_over_union, bb_intersection_over_union_np
from coordinate_change import xywh_tlbr, tlbr_xywh
from TP_TN_FP_FN import *
from load_data import norm_train_max_min, load_pkl
from load_data_binary import compute_iou




def anomaly_metric(prob, avg_or_max, pred_trajs, gt_box, vid_loc, frame_loc, person_id, abnormal_gt, abnormal_person):
    """
    This functon helps calculate abnormality by averaging or taking the max of pedestrains
    that belong to the same frame.

    
    """
    vid_frame = np.append(vid_loc, frame_loc, axis=1)
    vid_frame_person = np.append(vid_frame, person_id, axis =1)

    unique, unique_inverse, unique_counts = np.unique(vid_frame_person, axis=0, return_inverse=True, return_counts=True)

    repeat_inverse_id = np.where(unique_counts >= 1)[0] 
    # makes sense cuz sometimes might be one pedestrain if the start off at  the front of seq


    calc_prob, frame, vid, id_y, gt_abnormal, std = [], [], [], [], [],[]
    std_iou_or_l2, bbox_list, abnormal_ped, gt = [], [], [], []
    for i in repeat_inverse_id:
        same_vid_frame_person = np.where(unique_inverse == i)[0]

        temp_prob = prob[same_vid_frame_person]
        temp_pred_trajs = pred_trajs[same_vid_frame_person]
        gt_box_temp = gt_box[same_vid_frame_person][0]

        if avg_or_max == 'avg':
            calc_prob.append(np.mean(prob[same_vid_frame_person]))
            bbox = temp_pred_trajs
            # bbox = np.mean(temp_pred_trajs, axis = 0) # For averageing
                
        if avg_or_max == 'max':
            calc_prob.append(np.max(prob[same_vid_frame_person]))
            max_loc = np.where( temp_prob == np.max(temp_prob))[0]

            if len(max_loc) > 1:
                bbox = temp_pred_trajs[max_loc[0]]

            else:
                bbox = temp_pred_trajs[max_loc]
               


        frame.append(vid_frame_person[same_vid_frame_person[0]][1])
        vid.append(vid_frame_person[same_vid_frame_person[0]][0])
        id_y.append(vid_frame_person[same_vid_frame_person[0]][2])
        gt_abnormal.append(abnormal_gt[same_vid_frame_person[0]])
        # print('std axis = 0 probably doesnt make sense for the vertical direction so might need to change')
        std.append( np.std(pred_trajs[same_vid_frame_person].reshape(-1,4), axis=0) ) # Might need to change this for the vertical direction
        # if hyparams['metric'] == 'iou':
        std_iou_or_l2.append(np.std(prob[same_vid_frame_person]))
        bbox_list.append(np.array(bbox).reshape(-1,4))
        abnormal_ped.append(abnormal_person[same_vid_frame_person][0]) #same person
        gt.append(np.array(gt_box_temp).reshape(-1))

        print('I dont think the gt bbox shape is correct')
        quit()


    out = {}
    out['prob'] = np.array(calc_prob).reshape(-1,1)
    out['frame'] = np.array(frame).reshape(-1,1)
    out['vid'] = np.array(vid).reshape(-1,1)
    out['id_y'] = np.array(id_y).reshape(-1,1)
    out['abnormal_gt_frame_metric'] = np.array(gt_abnormal).reshape(-1,1)
    out['std'] = np.array(std).reshape(-1,4)
    out['std_iou_or_l2'] = np.array(std_iou_or_l2).reshape(-1,1)
    out['bbox'] = xywh_tlbr(np.array(bbox_list, dtype=object)) #  Note that this can be diff shapes in diff index
    out['abnormal_ped_pred'] = np.array(abnormal_ped).reshape(-1,1)
    out['gt_bbox'] = xywh_tlbr(np.array(gt))


    return out


def iou_as_probability(testdicts, models, max1 =None, min1 =None):
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
    if models =='bitrap':
        iou = []
        for testdict in testdicts:
            y = testdict['y_ppl_box'] # this is the gt 
            # gt_bb_unorm_tlbr = xywh_tlbr(np.squeeze(y))
            gt_bb_unorm_tlbr = xywh_tlbr(y)
            predicted_bb_unorm_tlbr = xywh_tlbr(testdict['pred_trajs'])
            iou = bb_intersection_over_union_np(    predicted_bb_unorm_tlbr,
                                                    gt_bb_unorm_tlbr )
            # need to squeeze to index correctly 
            iou.append(np.squeeze(iou))
        
        iou  = np.concatenate(iou)

    else:
        print('CHECK CODE BELOW TO MAKE SURE THEN REMOVE THE QUIT')
        quit()
        iou = []
        for testdict, model in zip(testdicts, models):
        # Need to fix this for lstm network
            x,y = norm_train_max_min(   testdict,
                                        # max1 = hyparams['max'],
                                        # min1 = hyparams['min']
                                        max1 = max1,
                                        min1 = min1
                                        )

            iou.append(compute_iou(x, y, max1, min1,  model))
            iou = np.concatenate(iou)

    iou_prob =  1 - iou

    return  iou_prob.reshape(-1,1)



def l2_error(testdicts, models, errortype, max1 = None, min1 =None ):     
    """
    testdicts: allows for multitime combinations
    models: allows for multitime combinations of lstm
    errortype: 'error_diff', 'error_summed', 'error_flattened'
    max1: hyperparameter
    min1: hyperparameter
    """
    preds = []
    if models == 'bitrap':
        for testdict in testdicts:
            preds.append(xywh_tlbr(testdict['pred_trajs']))

        
    else:
        for testdict, model in zip(testdicts, models):
            x,y = norm_train_max_min( testdict, max1 = max1, min1 = min1)
            shape =  x.shape
            pred = model.predict(x)

            preds.append(xywh_tlbr(pred.reshape(shape[0] ,-1,4)))

    
    trues = []
    for testdict in testdicts:
        trues.append(xywh_tlbr(testdict['y_ppl_box']))
    
    summed_list = []
    for pred, true in zip(preds, trues):
        diff = (true-pred)**2
        summed = np.sum(diff, axis =2)

        if errortype == 'error_diff':
            error = np.sum(np.diff(summed, axis =1), axis=1)
        elif errortype == 'error_summed':
            error = np.sum(summed, axis=1)
        elif errortype == 'error_flattened':
            error = summed.reshape(-1,1)

        summed_list.append(error)


        
        

    
    l2_error  = np.concatenate(summed_list)
    mmin = np.min(summed)
    mmax = np.max(summed)

    
    l2_norm = (l2_error - mmin)/ (mmax - mmin)


    return l2_norm.reshape(-1,1)



def combine_time(testdicts, errortype, models='bitrap', max1 =None, min1 = None):
    
    
    vid_loc, person_id, frame_loc, abnormal_gt_frame, abnormal_event_person, gt_bbox, pred_trajs = [],[],[],[],[], [],[]

    for testdict in testdicts:
        if errortype =='error_diff' or errortype == 'error_summed':
            person_id.append(testdict['id_y'][:,0] )
            frame_loc.append(testdict['frame_y'][:,0] )# frame locations
            abnormal_gt_frame.append(testdict['abnormal_gt_frame'][:,0])
            abnormal_event_person.append(testdict['abnormal_ped_pred'][:,0])
            gt_bbox.append(testdict['y_ppl_box'])
            print('Note that this does not work fully yet for testdicts in terms of gt_bboxes')
            vid_loc = testdict['video_file'].reshape(-1,1) #videos locations


        else:
            person_id.append(testdict['id_y'].reshape(-1,1))
            frame_loc.append(testdict['frame_y'].reshape(-1,1) )# frame locations
            abnormal_gt_frame.append(testdict['abnormal_gt_frame'].reshape(-1,1))
            abnormal_event_person.append(testdict['abnormal_ped_pred'].reshape(-1,1))
            gt_bbox.append(testdict['y_ppl_box'].reshape(-1,4))
            temp_vid_loc = testdict['video_file'].reshape(-1,1) #videos locations

        
            for vid in temp_vid_loc:
                vid_loc.append(np.repeat(vid,testdict['y_ppl_box'].shape[1]))


        if models =='bitrap':
            if errortype =='error_diff' or errortype == 'error_summed':
                pred_trajs.append(testdict['pred_trajs'] )
            elif errortype == 'error_flattened':
                pred_trajs.append(testdict['pred_trajs'].reshape(-1,4) )
        else:
            for testdict, model in zip(testdicts, models):
                x,_ = norm_train_max_min( testdict, max1 = max1, min1 = min1)
                predicted_bb = model.predict(x)
                predicted_bb_unorm = norm_train_max_min(predicted_bb, max1, min1, True)

                if errortype =='error_diff' or errortype == 'error_summed':
                    pred_trajs.append(predicted_bb_unorm)

                    print('check if the size of pred_tras is correct')
                    quit()

                elif errortype =='error_flattened':
                    pred_trajs.append(predicted_bb_unorm.reshape(-1,4) )# This is in xywh
            


    # Initilzation 
    pkldict = {}
    pkldict['id_y'] = np.concatenate(person_id).reshape(-1,1)
    pkldict['frame_y'] = np.concatenate(frame_loc).reshape(-1,1)
    pkldict['abnormal_gt_frame'] = np.concatenate(abnormal_gt_frame).reshape(-1,1)
    pkldict['abnormal_ped_pred'] = np.concatenate(abnormal_event_person).reshape(-1,1)
    pkldict['vid_loc'] = np.concatenate(vid_loc).reshape(-1,1)
    pkldict['gt_bbox'] = np.concatenate(gt_bbox)
    pkldict['pred_trajs'] = np.concatenate(pred_trajs)


    return pkldict

if __name__ == '__main__':

    pkldicts = []
    pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(20,10)))
    # pkldicts.append(load_pkl(loc['pkl_file']['avenue_template'].format(5,5)))
    
    # pkldict = load_pkl(loc['pkl_file']['avenue_template'].format())
    

    error = combine_time(pkldicts, errortype='error_summed', models = 'bitrap')

    # error = l2_error(pkldicts, 'bitrap' , 'error_flattened')

