# This is in develpment rn
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys, time
from os.path import join


from config import hyparams, loc, exp

from custom_metrics import bb_intersection_over_union, bb_intersection_over_union_np
from coordinate_change import xywh_tlbr, tlbr_xywh
from TP_TN_FP_FN import *






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
        # gt_bb_unorm_tlbr = xywh_tlbr(np.squeeze(y))
        gt_bb_unorm_tlbr = xywh_tlbr(y)
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


def l2(testdict, model):

    if model == 'bitrap':
        true = testdict['y_ppl_box']
        pred = testdict['pred_trajs']
        diff = true - pred
        summed = np.sum( diff, axis = 2)

    return summed
