import os, sys, time
from os.path import join
import numpy as np

# Plotting Metrics
from sklearn.metrics import roc_curve, auc

#Currentfolder
from custom_functions.coordinate_change import xywh_tlbr
from custom_functions.metrics_plot import generate_metric_plots
from custom_functions.visualizations import plot_frame_from_image, plot_vid, generate_images_with_bbox
from custom_functions.utils import make_dir, SaveTextFile, SaveAucTxt, SaveAucTxtTogether
from custom_functions.visualizations import plot_sequence
from custom_functions.auc_metrics import iou_as_probability, l2_error, giou_ciou_diou_as_metric
# From different folders
from config.config import hyparams, loc, exp


def frame_traj_model_auc(model, testdicts, metric, avg_or_max, modeltype, norm_max_min):
    """
    This function is meant to find the frame level based AUC
    model: any trajactory prediction model (would need to check input matches)
    testdicts: is the input data dict
    metric: iou or l2, giou, ciou,diou metric
    avg_or_max: used when looking at same person over frame, for vertical comparssion

    return:
    auc_human_frame: human and frame level auc
    """

    # Note that this return ious as a prob 
    test_auc_frame, remove_list, out_frame, out_human = ped_auc_to_frame_auc_data(model, testdicts, metric, avg_or_max, modeltype, norm_max_min)

    # Plot abnormal Graph scores
    # plot_frame_wise_scores(out_frame)
    
    nc = [  loc['nc']['date'] + '_per_frame',
            loc['nc']['model_name'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['input_seq'],
            hyparams['pred_seq']
            ] # Note that frames is the sequence input
    
    if test_auc_frame == 'not possible':
        quit()
    
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

    # This plots the result of bbox in the images 
    if exp['plot_images']:
        generate_images_with_bbox(testdicts,out_frame, visual_path)

    # generate_metric_plots(test_auc_frame, metric, nc, plot_loc)   
    # else:
    #     file_avg_metrics = SaveTextFile(plot_loc, metric)
    #     file_avg_metrics.save(output_with_metric, auc_frame_human)
    #     file_with_auc = SaveAucTxt(joint_txt_file_loc, metric)
    #     file_with_auc.save(auc_frame_human)

    #     print(joint_txt_file_loc)

    y_true = test_auc_frame['y']
    y_pred = test_auc_frame['x']
    y_true_per_human = test_auc_frame['y_pred_per_human']
    y_pred_per_human = test_auc_frame['x_pred_per_human']


    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    AUC_frame = auc(fpr, tpr)
    fpr, tpr, thresholds = roc_curve(y_true_per_human, y_pred_per_human)
    AUC_human = auc(fpr,tpr)
    auc_human_frame = np.array([AUC_human, AUC_frame])
    
    return auc_human_frame



def ped_auc_to_frame_auc_data(model, testdicts, metric, avg_or_max, modeltype, norm_max_min):
    """
    removes select points to reduce to frame AUC
    input:
        testdict: From orginal data test dict
        model: lstm model
        metric: l2, giou,l2
        modeltype: type of model
    
    return:
        test_auc_frame: people that are in the frame as proability and index of frame
        remove_list: indices of pedestrain removed
    """
    # used i-iou, 1-giou, 1-ciou, 1-diou etc 
    if metric == 'iou':
        prob, prob_along_time = iou_as_probability(testdicts, model, errortype = hyparams['errortype'], max1 = norm_max_min.max, min1= norm_max_min.min)

    elif metric == 'l2':
        
        prob, prob_along_time = l2_error(testdicts = testdicts, models = model, errortype =hyparams['errortype'], max1=norm_max_min.max , min1= norm_max_min.min)

    elif metric == 'giou' or metric =='ciou' or metric =='diou':
        prob, prob_along_time = giou_ciou_diou_as_metric(testdicts = testdicts, models = model, metric=metric,errortype = hyparams['errortype'], max1 = norm_max_min.max,min1 = norm_max_min.min)


    # Creates a represention to combine multiple time sequences  
    pkldicts = combine_time(    testdicts, models=model, errortype=hyparams['errortype'], 
                                modeltype = modeltype,  max1 = norm_max_min.max, min1= norm_max_min.min)


    # Finds pedestrains same vid_frame_id and avgs
    out = anomaly_metric(   prob, 
                            avg_or_max,
                            pkldicts,
                            prob_in_time = prob_along_time
                        )
    
    prob = out['prob'] 
    vid_loc = out['vid']
    frame_loc = out['frame']
    
    # frame_loc, vid_loc, abnormal_gt_frame_metric, std, std_iou
    test_index = np.arange(0, len(out['abnormal_gt_frame_metric']), 1)
    ##############
    
    # encoding video and frame 
    vid_frame = np.append(vid_loc, frame_loc, axis=1)

    unique, unique_inverse, unique_counts = np.unique(vid_frame, axis=0, return_inverse=True, return_counts=True)
    repeat_inverse_id = np.where(unique_counts>1)[0] 

    
    # Pedestrain AUC equals Frame AUC
    if len(repeat_inverse_id) == 0:
        test_auc_frame = 'not possible'
     
    # Convert Pedestrain AUC to Frame AUC
    else:
        remove_list_temp = []
        for i in repeat_inverse_id:
            # find all same vid and frame
            same_vid_frame = np.where(unique_inverse == i)[0]
            y_pred = prob[same_vid_frame]

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

        test_auc_frame = {}
        test_auc_frame['x'] = np.delete( prob.reshape(-1,1), remove_list, axis = 0 )
        test_auc_frame['y'] = np.delete(out['abnormal_gt_frame_metric'], remove_list, axis = 0)
        test_auc_frame['x_pred_per_human'] = prob.reshape(-1,1)
        test_auc_frame['y_pred_per_human'] = out['abnormal_gt_frame_metric']
        test_auc_frame['std_per_frame'] = np.delete(out['std'], remove_list, axis = 0)
        test_auc_frame['std_per_human'] = out['std']
        test_auc_frame['index'] = test_index.reshape(-1,1)

        test_auc_frame['std_iou_or_l2_per_frame'] = np.delete(out['std_iou_or_l2'], remove_list, axis = 0)
        test_auc_frame['std_iou_or_l2_per_human'] = out['std_iou_or_l2']
                                         
        out_frame = {}
        for key in out.keys():
            out_frame[key] = np.delete(out[key], remove_list, axis = 0)

        out_human = out
    return test_auc_frame, remove_list, out_frame, out_human




def anomaly_metric(prob, avg_or_max, pkldicts, prob_in_time):
    """
    This functon helps calculate abnormality by averaging or taking the max of pedestrains
    that belong to the same frame and video.

    This is allows for aggravating of pedestrains together     
    
    Note if using error_diff or error_summed,
    because we use the first frame of the trajectory as represention (in combine_time). 
    Can be thought of as making overlapping trajectory nonoverlapping 
    therefore there is max/avg does not affect probs 
    
    
    """
    
    pred_trajs = pkldicts['pred_trajs']
    gt_box = pkldicts['gt_bbox']
    vid_loc = pkldicts['vid_loc']
    frame_loc = pkldicts['frame_y']
    person_id = pkldicts['id_y']
    abnormal_gt = pkldicts['abnormal_gt_frame']
    abnormal_person = pkldicts['abnormal_ped_pred']
    
    dim = pred_trajs.shape[1]
    vid_frame = np.append(vid_loc, frame_loc, axis=1)
    vid_frame_person = np.append(vid_frame, person_id, axis =1)

    unique, unique_inverse, unique_counts = np.unique(vid_frame_person, axis=0, return_inverse=True, return_counts=True)

    repeat_inverse_id = np.where(unique_counts >= 1)[0] 
    # makes sense cuz sometimes might be one pedestrain if the start off at  the front of seq

    calc_prob, frame, vid, id_y, gt_abnormal, std = [], [], [], [], [],[]
    std_iou_or_l2, bbox_list, abnormal_ped, gt = [], [], [], []
    prob_with_time = []
    for i in repeat_inverse_id:
        same_vid_frame_person = np.where(unique_inverse == i)[0] # keeps indices aligned 
        temp_prob = prob[same_vid_frame_person]
        temp_pred_trajs = pred_trajs[same_vid_frame_person] # can have diff shapes if error_flattend
        gt_box_temp = gt_box[same_vid_frame_person] # can have diff shapes if eror_flattened

        if avg_or_max == 'avg':
            calc_prob.append(np.mean(prob[same_vid_frame_person]))
            bbox = temp_pred_trajs
            # bbox = np.mean(temp_pred_trajs, axis = 0) # For averageing
            if hyparams['errortype']=='error_flattened':
                prob_with_time.append(-1)
            else:    
                prob_with_time.append(prob_in_time[same_vid_frame_person])
                
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
        std.append( np.std(pred_trajs[same_vid_frame_person].reshape(-1,dim), axis=0) )
        std_iou_or_l2.append(np.std(prob[same_vid_frame_person]))
        abnormal_ped.append(abnormal_person[same_vid_frame_person][0]) #same person
        bbox_list.append(np.array(bbox).reshape(-1,dim))
        gt.append(np.array(gt_box_temp).reshape(-1,dim))


    out = {}
    out['prob'] = np.array(calc_prob).reshape(-1,1)
    out['frame'] = np.array(frame, dtype=int).reshape(-1,1)
    out['vid'] = np.array(vid).reshape(-1,1)
    out['id_y'] = np.array(id_y, dtype =int).reshape(-1,1)
    out['abnormal_gt_frame_metric'] = np.array(gt_abnormal).reshape(-1,1)
    out['std'] = np.array(std).reshape(-1,dim)
    out['std_iou_or_l2'] = np.array(std_iou_or_l2).reshape(-1,1)
    out['abnormal_ped_pred'] = np.array(abnormal_ped)#.reshape(-1,1)
    if exp['pose']:
        out['pred_bbox'] = np.array(bbox_list, dtype=object) #  Note that this can be diff shapes in diff index
        out['gt_bbox'] = np.array(gt)
    else:
        out['pred_bbox'] = xywh_tlbr(np.array(bbox_list, dtype=object)) #  Note that this can be diff shapes in diff index
        out['gt_bbox'] = xywh_tlbr(np.array(gt))
    
    out['prob_with_time'] = np.squeeze( np.array(prob_with_time) )
        
    return out













def combine_time(testdicts, models, errortype, modeltype, max1 =None, min1 = None):
    """
    This function gives ablity to combine multiple different timescales. Meaning that can
    have multiple testdict from different timescales and then extract the first frame to represent that
    trajectory in  error_summed or error_diff setting. This sets the the anomaly metric function
    by allowing "compressing" input into first y frame with video number. 
    
    For error_flattend method, combine time allows for different dicts to be flattened,
    which helps down the line in anomaly_metric function (find pedestrains from the same frame to average errors.)

    output:
        pkldict with keys:
            pkldict['id_y'] 
            pkldict['frame_y']
            pkldict['abnormal_gt_frame'] 
            pkldict['abnormal_ped_pred'] 
            pkldict['vid_loc']
            pkldict['gt_bbox']
            pkldict['pred_trajs']

    """
    
    vid_loc, person_id, frame_loc, abnormal_gt_frame, abnormal_event_person, gt_bbox, pred_trajs = [],[],[],[],[], [],[]
    dim = testdicts[0]['y_ppl_box'].shape[-1]
    for testdict in testdicts:
        if errortype =='error_diff' or errortype == 'error_summed':
            person_id.append(testdict['id_y'][:,0] )
            frame_loc.append(testdict['frame_y'][:,0] )
            abnormal_gt_frame.append(testdict['abnormal_gt_frame'][:,0])
            # abnormal_event_person.append(testdict['abnormal_ped_pred'][:,0])
            abnormal_event_person.append(testdict['abnormal_ped_pred'])
            gt_bbox.append(testdict['y_ppl_box'])
            vid_loc = testdict['video_file'].reshape(-1,1) #videos locations

        else:
            person_id.append(testdict['id_y'].reshape(-1,1))
            frame_loc.append(testdict['frame_y'].reshape(-1,1) )
            abnormal_gt_frame.append(testdict['abnormal_gt_frame'].reshape(-1,1))
            abnormal_event_person.append(testdict['abnormal_ped_pred'].reshape(-1,1))
            gt_bbox.append(testdict['y_ppl_box'].reshape(-1,dim))
            temp_vid_loc = testdict['video_file'].reshape(-1,1) #videos locations

        
            for vid in temp_vid_loc:
                vid_loc.append(np.repeat(vid,testdict['y_ppl_box'].shape[1]))


        if modeltype =='bitrap':
            if errortype =='error_diff' or errortype == 'error_summed':
                pred_trajs.append(testdict['pred_trajs'] )
            elif errortype == 'error_flattened':
                pred_trajs.append(testdict['pred_trajs'].reshape(-1,dim) )
        else:
            for testdict, model in zip(testdicts, models):
                # Might be good to replace this
                x,_ = norm_train_max_min( testdict, max1 = max1, min1 = min1)
                shape =  x.shape
                predicted_bb = model.predict(x)
                predicted_bb_unorm_xywh = norm_train_max_min(predicted_bb, max1, min1, True)
                predicted_bb_unorm_xywh = predicted_bb_unorm_xywh.reshape(shape[0] ,-1,4)
                ########################

                if errortype =='error_diff' or errortype == 'error_summed':
                    pred_trajs.append(predicted_bb_unorm_xywh)

                    # print('check if the size of pred_tras is correct')
                    # quit()

                elif errortype =='error_flattened':
                    pred_trajs.append(predicted_bb_unorm_xywh.reshape(-1,4) )# This is in xywh
            


    # Initilzation 
    pkldict = {}
    pkldict['id_y'] = np.concatenate(person_id).reshape(-1,1)
    pkldict['frame_y'] = np.concatenate(frame_loc).reshape(-1,1)
    pkldict['abnormal_gt_frame'] = np.concatenate(abnormal_gt_frame).reshape(-1,1)
    # pkldict['abnormal_ped_pred'] = np.concatenate(abnormal_event_person).reshape(-1,1)
    pkldict['abnormal_ped_pred'] = np.concatenate(abnormal_event_person)
    pkldict['vid_loc'] = np.concatenate(vid_loc).reshape(-1,1)
    pkldict['gt_bbox'] = np.concatenate(gt_bbox)
    pkldict['pred_trajs'] = np.concatenate(pred_trajs)


    return pkldict