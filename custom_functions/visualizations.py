import cv2
import os
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
from config.config import hyparams, loc, exp
from custom_functions.utils import make_dir

## For __name___ = __main__
from custom_functions.load_data import  load_pkl
from custom_functions.anomaly_detection import ped_auc_to_frame_auc_data
from config.max_min_class_global import Global_Max_Min

RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# def generate_images_with_bbox(testdicts,out_frame, visual_path):
def generate_folders(out_frame, visual_path):
    visual_plot_loc = join( os.path.dirname(os.getcwd()), *visual_path )
    if exp['data'] =='avenue':
        for i in range(1,22):
            path = visual_path.copy()
            path.append('{:02d}'.format(i))
            make_dir(path)
            
            if not hyparams['errortype']=='error_flattened':

                path_timeseries = visual_path.copy()
                path_timeseries.append('{:02d}_time_series'.format(i))
                make_dir(path_timeseries)


    elif exp['data']=='st':
        for txt in np.unique(out_frame['vid']):
            path = visual_path.copy()
            path.append('{}'.format(txt[:-4]))
            make_dir(path)

            if not hyparams['errortype']=='error_flattened':
                path_timeseries = visual_path.copy()
                path_timeseries.append('{}_time_series'.format(txt[:-4]))
                make_dir(path_timeseries)

    


def plot_frame_from_image(pic_loc, bbox_preds, save_to_loc, vid, frame, idy, prob,abnormal_frame, abnormal_ped, gt_bboxs = None):
    """
    pic_loc: this is where the orginal pic is saved
    bbox_pred: this is where the bbox is saved
    """
    img = cv2.imread(pic_loc)
    for bbox_pred in bbox_preds:
        # cv2.rectangle(img, (int(bbox_pred[0]), int(bbox_pred[1])), (int(bbox_pred[2]), int(bbox_pred[3])),(0,255,255), 2)
        cv2.rectangle(img, (int(bbox_pred[0]), int(bbox_pred[1])), (int(bbox_pred[2]), int(bbox_pred[3])), WHITE , 2)
    
    for gt_bbox in gt_bboxs:
        # Doing this way also takes care of multiple bounding boxes
        cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])),(0,255,0), 2)


    # cv2.putText(img, '{:.4f}'.format(prob),(25,65),0, 5e-3 * 100, (255,255,0),2)
    
    # # This is for abnormal frame, this uses the last bbox of set to plot 
    # if abnormal_frame:
    #     cv2.putText(img, 'Abnormal Frame',(25,25),0, 5e-3 * 150, (0,0,255),2)
    # else:
    #     cv2.putText(img, 'Normal Frame',(25, 25),0, 5e-3 * 150, (0,255, 0 ),2)
    
    # # This is for abnormal person
    # if abnormal_ped:
    #     cv2.putText(img, '1',(25,45),0, 5e-3 * 100, (0,0,255),2)

    # else:
    #     cv2.putText(img, '0',(25,45),0, 5e-3 * 100, (255,0, 0),2)



    # if abnormal_frame and abnormal_ped:
    
    cv2.imwrite(save_to_loc + '/' + '{}'.format(vid) + '/' + '{}__{}_{}_{:.4f}.jpg'.format(vid, frame, idy, prob), img)


def plot_error_in_time(prob_with_time, time_series_plot_loc, vid, frame, idy, is_traj_abnormal=False):
    fig,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(prob_with_time, '-*') #, label='Error summed')
    ax.set_xlabel('Time')
    ax.set_ylabel('Error Summed')
    ax.set_title('video:{} frame:{} idy:{}'.format(vid, frame, idy ))
    # ax.legend()

    if is_traj_abnormal:
        plot_name = 'vid_{}_sframe_{:02d}_id_{:02d}_abnorm.jpg'.format(vid,frame,idy)
    else:
        plot_name = 'vid_{}_sframe_{:02d}_id_{:02d}.jpg'.format(vid,frame,idy)
                
    img_path = join( time_series_plot_loc, '{}_time_series'.format(vid), plot_name)

    fig.savefig(img_path)
    plt.close(fig)

def plot_vid(out_frame, pic_locs, visual_plot_loc, data):

    """
    out_frame: this is the dict
    pic_loc: pic that will be plottd
    visual_plot_loc: this is where the video will be saved at
    """
    for bbox_preds, vid, frame, idy, prob, prob_with_time, abnormal_frame, abnormal_ped, gt_bbox in zip(    out_frame['pred_bbox'], out_frame['vid'], 
                                                                                                            out_frame['frame'], out_frame['id_y'],
                                                                                                            out_frame['prob'],
                                                                                                            out_frame['prob_with_time'],
                                                                                                            out_frame['abnormal_gt_frame_metric'],
                                                                                                            out_frame['abnormal_ped_pred'],
                                                                                                            out_frame['gt_bbox'] ):
        if data =='avenue':
            pic_loc = join(  pic_locs, '{:02d}'.format(int(vid[0][:-4])) )
            pic_loc =  pic_loc + '/' +'{:02d}.jpg'.format(int(frame))
        elif data =='st':
            pic_loc = join(  pic_locs, '{}'.format(vid[0][:-4]) )
            pic_loc =  pic_loc + '/' +'{:03d}.jpg'.format(int(frame))

        plot_frame_from_image(  pic_loc = pic_loc,  
                                bbox_preds = bbox_preds ,
                                save_to_loc = visual_plot_loc, 
                                vid = vid[0][:-4],
                                frame = int(frame[0]), 
                                idy = int(idy[0]),
                                prob = prob[0],
                                abnormal_frame = abnormal_frame,
                                abnormal_ped = abnormal_ped,
                                gt_bboxs = gt_bbox)
        if not hyparams['errortype']=='error_flattened':
            plot_error_in_time(prob_with_time, visual_plot_loc, vid[0][:-4], int(frame[0]), int(idy[0]))


def plot_sequence(person_seq, max1, min1,pic_locs, save_plot_loc, xywh=False):
    gt_bboxs = person_seq['gt_bbox']
    bbox_preds = person_seq['pred_bbox']

    if xywh:
        gt_bboxs[:,0]  =  gt_bboxs[:,0]  -  gt_bboxs[:,2]/2
        gt_bboxs[:,1]  =  gt_bboxs[:,1]  -  gt_bboxs[:,3]/2 # Now we are at tlwh
        gt_bboxs[:,2:] =  gt_bboxs[:,:2] +  gt_bboxs[:,2:]

        bbox_preds[:,0]  =  bbox_preds[:,0]  -  bbox_preds[:,2]/2
        bbox_preds[:,1]  =  bbox_preds[:,1]  -  bbox_preds[:,3]/2 # Now we are at tlwh
        bbox_preds[:,2:] =  bbox_preds[:,:2] +  bbox_preds[:,2:]
    frame = person_seq['frame'][0]

    for bbox_pred, gt_bbox in zip(bbox_preds, gt_bboxs):
        current_image_loc = join(pic_locs, '{:04d}.jpg'.format(frame))
        img = cv2.imread(current_image_loc)
        cv2.rectangle(img, (int(bbox_pred[0]), int(bbox_pred[1])), (int(bbox_pred[2]), int(bbox_pred[3])),WHITE, 2)
        cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])),(0,255,0), 2)
        cv2.imwrite(save_plot_loc + '/{:04d}.jpg'.format(frame), img)
        frame = frame + 1

if __name__ == '__main__':
    norm_max_min = Global_Max_Min()
    testdicts = [ load_pkl(loc['pkl_file'][exp['data']], exp['data']) ]
    test_auc_frame, remove_list, out_frame, out_human = ped_auc_to_frame_auc_data( 'bitrap', testdicts, hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], norm_max_min)
    visual_path = loc['visual_trajectory_list'].copy()
    visual_path.append('{}_{}_{}_in_{}_out_{}_{}'.format(  loc['nc']['date'],
                                                        hyparams['metric'],
                                                        hyparams['avg_or_max'], 
                                                        hyparams['input_seq'], 
                                                        hyparams['pred_seq'],
                                                        hyparams['errortype'] ) )

    # generate_folders(out_frame, visual_path)

    for i in range(0,out_frame['vid'].shape[0]):
        person_seq = {}
        for key in out_frame.keys():
            person_seq[key] = out_frame[key][i]
        

        is_traj_abnormal = np.array(person_seq['abnormal_ped_pred'])
        is_traj_abnormal = np.any(is_traj_abnormal==1)

        video_num = int(person_seq['vid'][0][:-4])
        frame = person_seq['frame'][0]
        idy = person_seq['id_y'][0]

        pic_locs = join(loc['data_load'][exp['data']]['pic_loc_test'],  '{:02d}'.format(video_num ) )
        save_img_path = '/mnt/roahm/users/akanu/projects/anomalous_pred/results_all_datasets/experiment_traj_model/visual_trajectory_consecutive_bitrap/02_17_22_in_{}_out_{}/frame/{:02d}'.format(hyparams['input_seq'], hyparams['pred_seq'], video_num)
        
        if is_traj_abnormal:
            path_list = [ 'vid_{:02d}_sframe_{:02d}_id_{:02d}_abnorm'.format( video_num, frame, idy ) ]
        else:
            path_list = [ 'vid_{:02d}_sframe_{:02d}_id_{:02d}'.format( video_num, frame, idy ) ]
        
        make_dir(path_list, save_img_path)
        save_plot_loc = join(save_img_path, *path_list)

        plot_sequence(  person_seq, 
                        norm_max_min.max,
                        norm_max_min.min,
                        pic_locs, 
                        save_plot_loc,
                        xywh=False)# Need to make sure pose checked off in config.py






if __name__ == '__main__':
    norm_max_min = Global_Max_Min()
    visual_path = loc['visual_trajectory_list'].copy()
    visual_path.append('{}_{}_{}_in_{}_out_{}_{}'.format(  loc['nc']['date'],
                                                        hyparams['metric'],
                                                        hyparams['avg_or_max'], 
                                                        hyparams['input_seq'], 
                                                        hyparams['pred_seq'],
                                                        hyparams['errortype'] ) )

    file_to_load = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/gaussian_avenue_in_{}_out_{}_K_1_pose_{}.pkl'.format(hyparams['input_seq'],
                                                                                                                                        hyparams['pred_seq'],
                                                                                                                                        ablation )

    model = 'bitrap'
    testdicts = [ load_pkl(file_to_load, exp['data']) ]
    test_auc_frame, remove_list, out_frame, out_human = ped_auc_to_frame_auc_data( 'bitrap', testdicts, hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], norm_max_min)
