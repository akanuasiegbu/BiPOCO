# This is in develpment rn
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys, time
from os.path import join


from config.config import hyparams, loc, exp

from coordinate_change import xywh_tlbr, tlbr_xywh
from load_data import  load_pkl
from custom_functions.iou_utils import giou, diou, ciou




def l2_norm_or_l2_error_weighted(testdicts, models, errortype, max1 = None, min1 =None, use_kp_confidence=False):     
    """
    testdicts: allows for multitime combinations
    models: allows for multitime combinations of lstm
    errortype: 'error_diff', 'error_summed', 'error_flattened'
    max1: hyperparameter
    min1: hyperparameter
    """
    preds = []
    kp_confidences = []
    if models == 'bitrap':
        for testdict in testdicts:
            if exp['pose']:
                preds.append( testdict['pred_trajs'] )
                if use_kp_confidence:
                    kp_confidences.append(testdict['kp_confidence_pred'])
            else:
                preds.append(xywh_tlbr(testdict['pred_trajs']))

        
    else:
        for testdict, model in zip(testdicts, models):
            x,y = norm_train_max_min( testdict, max1 = max1, min1 = min1)
            shape =  x.shape
            pred = model.predict(x)

            preds.append(xywh_tlbr(pred.reshape(shape[0] ,-1,4)))

    
    trues = []
    for testdict in testdicts:
        if exp['pose']:
            trues.append( testdict['y_ppl_box'] )
        else:
            trues.append(xywh_tlbr(testdict['y_ppl_box']))
    
    summed_list, output_in_time = [], []
    if use_kp_confidence:
        for pred, true, kp_confidence in zip(preds, trues, kp_confidences):
            error, out_in_time = l2_norm_or_l2_error_weighted_sub_calc(pred, true, errortype, kp_confidence, use_kp_confidence)    
            summed_list.append(error)
            output_in_time.append(out_in_time)

    else:
        for pred, true in zip(preds, trues):
            error, out_in_time = l2_norm_or_l2_error_weighted_sub_calc(pred, true, errortype)    
            summed_list.append(error)
            output_in_time.append(out_in_time)
    
    output_in_time = np.concatenate(output_in_time)
    l2_error  = np.concatenate(summed_list)

    return l2_error.reshape(-1,1), output_in_time

def  l2_norm_or_l2_error_weighted_sub_calc(pred, true, errortype, kp_confidence = None, use_kp_confidence=False):
    # Note currently masking out the bounding boxes during the calcuation
    # Note removing the bounding boxes in calcuation 
    diff = (true-pred)**2
    if use_kp_confidence:
        if true.shape[-1] == 38:
            kp_confidence = np.repeat(kp_confidence, 2, axis=2)
            kp_confidence_with_mask = np.pad(kp_confidence, ((0,0), (0,0), (0,4)))
        if true.shape[-1] == 36:
            kp_confidence = np.repeat(kp_confidence, 2, axis=2)
            kp_confidence_with_mask = np.pad(kp_confidence, ((0,0), (0,0), (0,2)))
        
        diff = diff*kp_confidence_with_mask
        summed = np.sum(diff, axis =2)

    if not use_kp_confidence:
        if true.shape[-1] ==38:
            summed = np.sum(diff[:,:,:34], axis =2) 
            summed = np.sqrt(summed)
        elif true.shape[-1]==36:
            summed = np.sum(diff, axis =2) 
            summed = np.sqrt(summed)


    if errortype == 'error_diff':
        error = np.sum(np.diff(summed, axis =1), axis=1)
    elif errortype == 'error_summed':
        error = np.sum(summed, axis=1)
    elif errortype == 'error_flattened':
        error = summed.reshape(-1,1)
    else:
        pass

    return error, summed




def oks_similarity(testdicts, models, metric, errortype, max1=None, min1=None):
    
    kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    kpt = []
    for i in kpt_oks_sigmas:
        kpt.append(i)
        kpt.append(i)
    kpt_oks_sigmas = np.array(kpt)

    preds = []
    kp_confidences = []
    if models == 'bitrap':
        for testdict in testdicts:
            if exp['pose']:
                preds.append( testdict['pred_trajs'] )
            else:
                preds.append(xywh_tlbr(testdict['pred_trajs']))


    trues = []
    for testdict in testdicts:
        if exp['pose']:
            trues.append( testdict['y_ppl_box'] )
        else:
            trues.append(xywh_tlbr(testdict['y_ppl_box']))
    oks_s = []
    for pred, true in zip(preds, trues):
        # oks calc
        diff = (true-pred)**2
        diff = diff[:,:,:34] 
        scale = calc_s(pred)
        exp_term = np.exp(-diff/(2*(scale**2)*(kpt_oks_sigmas**2) +0.00001))
        
        oks_s.append(np.sum(exp_term, axis =2))

    oks_s = np.concatenate(oks_s)
        # print(ans[0][0])
    if errortype == 'error_diff':
        output = np.sum(np.diff(oks_s, axis =1), axis=1)
    elif errortype == 'error_summed':
        output = np.sum(oks_s, axis=1)
    elif errortype == 'error_flattened':
        output = oks_s.reshape(-1,1)
    return output, oks_s

def calc_s(keypoints):
    # keypoints shape: (traj_num, pred_len, keypoints)
    # if exp['data'] == 'avenue':
    if 'avenue' in exp['data']:
        video_resolution = np.array([640, 360], dtype= np.float32)
    
    tlbr = compute_bounding_box(keypoints, video_resolution, return_discrete_values=True)   
    area = calc_area(tlbr)
    return area


def compute_bounding_box(keypoints, video_resolution, return_discrete_values=True):
    """Compute the bounding box of a set of keypoints.
    Took from MPED-RNN Code and vectorized 
    Morais et al. 
    Argument(s):
        keypoints -- A numpy array, of shape (num_keypoints * 2,), containing the x and y values of each
            keypoint detected.
        video_resolution -- A numpy array, of shape (2,) and dtype float32, containing the width and the height of
            the video.

    Return(s):
        The bounding box of the keypoints represented by a 4-uple of integers. The order of the corners is: left,
        right, top, bottom.
    """
    width, height = video_resolution
    # keypoints_reshaped = keypoints.reshape(-1, 2)
    keypoints_reshaped = keypoints
    x, y = keypoints_reshaped[:,:, 0::2], keypoints_reshaped[:,:,1::2]
    # x, y = x[x != 0.0], y[y != 0.0]
    try:
        left, right, top, bottom = np.min(x, axis=-1), np.max(x, axis=-1), np.min(y, axis =-1), np.max(y, axis =-1)
    except ValueError:
        # print('All joints missing for input skeleton. Returning zeros for the bounding box.')
        return 0, 0, 0, 0

    extra_width, extra_height = 0.1 * (right - left + 1), 0.1 * (bottom - top + 1)
    left, right = np.clip(left - extra_width, 0, width - 1), np.clip(right + extra_width, 0, width - 1)
    top, bottom = np.clip(top - extra_height, 0, height - 1), np.clip(bottom + extra_height, 0, height - 1)
    
    left = np.expand_dims(left, axis =2 )
    right = np.expand_dims(right, axis =2 )
    top = np.expand_dims(top, axis=2)
    bottom = np.expand_dims(bottom, axis=2)
    
    tlbr = np.concatenate((left,top, right, bottom), axis=2)
    if return_discrete_values:
        return tlbr.round().astype(int)
    else:
        return tlbr



def calc_area(tlbr):
    tlbr = tlbr.astype(float)
    area = (tlbr[:,:, 2:3] - tlbr[:,:,0:1])*(tlbr[:,:,3:4] - tlbr[:,:,1:2])
    return area

if __name__ == '__main__':

    file_to_load = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/gaussian_avenue_in_{}_out_{}_K_1_pose_hc.pkl'.format(hyparams['input_seq'],
                                                                                                                                        hyparams['pred_seq']
                                                                                                                                         )

    testdicts = [ load_pkl(file_to_load, exp['data']) ]
    oks_similarity(testdicts, models='bitrap', metric='none', errortype='none', max1=None, min1=None)

    # keypoints = testdicts[0]['pred_trajs']
    # video_resolution = np.array([640, 360], dtype= np.float32)
    # tlbr = compute_bounding_box(keypoints, video_resolution, return_discrete_values=True)
    # area = calc_area(tlbr)


