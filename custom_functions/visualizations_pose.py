# Copied from AlphaPose vis.py
import cv2
import numpy as np
import os
import math

from custom_functions.load_data import load_pkl
from custom_functions.utils import make_dir
from custom_functions.ped_sequence_plot import ind_seq_dict  
from config.config import hyparams, loc, exp


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

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]

    return color

class Opt():
    showbox = False
    tracking = True
def vis_frame(frame, im_res, opt, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    kp_num = 17
    if len(im_res['result']) > 0:
    	kp_num = len(im_res['result'][0]['keypoints'])

    
    if kp_num == 17 or kp_num == 18:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]

            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        
    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    index = 0
    
    # Labels normal and abnormal frames
    if im_res['result'][0]['abnormal_gt_frame'] == 1:
        cv2.putText(img, 'Abnormal Frame',(25,25),0, 5e-3 * 150, (0,0,255),2)
    else:
        cv2.putText(img, 'Normal Frame',(25,25),0, 5e-3 * 150, (0,255,0),2)


    for human in im_res['result']:
        
        part_line = {}
        kp_preds = human['keypoints']
        if kp_num == 17:
            kp_preds = np.concatenate((kp_preds, np.expand_dims((kp_preds[5,:] + kp_preds[6,:]) / 2, axis=0) ) ) # added to match alphapose vis_fast
        if opt.tracking:
            color = get_color_fast(int(abs(human['idx'])))
        else:
            if index == 0:
                color = WHITE #using blue to show input and white for predictions
                # color = YELLOW #using blue to show input and white for predictions
                # index += 1
            elif index == 1:
                color = GREEN # Ground truth
                # continue
            index += 1

        # Draw bboxes
        if opt.showbox:
            if 'box' in human.keys():
                bbox = human['box']
                bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]#xmin,xmax,ymin,ymax # tlwh to tlbr (alphapose format)
                # bbox = [bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2, bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]   # midx_midy,w,h to tlbr (bitrap format)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)
            if human['abnormal_ped_pred'] == 1 and index == 0:
            # if human['abnormal_pedestrain'] == 1:
                cv2.putText(img, str(human['idx']) + 'ABNORM', (int(bbox[0]), int((bbox[1] ))), DEFAULT_FONT, 1, color, 2)
            elif index == 0 :
                cv2.putText(img, str(human['idx'] ), (int(bbox[0]), int((bbox[1]))), DEFAULT_FONT, 1, color, 2)

        # # Draw keypoints
        for n in range(18):
            # if kp_scores[n] <= vis_thres:
            #     continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x), int(cor_y))
            bg = img.copy()
            if n < len(p_color):
                if opt.tracking:
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, color, -1)
                else:
                    # cv2.circle(bg, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
                    cv2.circle(bg, (int(cor_x), int(cor_y)), 2, color, -1)
            else:
                cv2.circle(bg, (int(cor_x), int(cor_y)), 1, (255,255,255), 2)
            # Now create a mask of logo and create its inverse mask also
            if n < len(p_color):
                # transparency = float(max(0, min(1, kp_scores[n])))
                transparency = 0.85
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
            # cv2.putText is temp
            # cv2.putText(img, str(n), (int(cor_x), int((cor_y))), DEFAULT_FONT, .2, GREEN, 1)
        
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length/2), int(stickwidth)), int(angle), 0, 360, 1)
                if i < len(line_color):
                    if opt.tracking:
                        cv2.fillConvexPoly(bg, polygon, color)
                    else:
                        cv2.fillConvexPoly(bg, polygon, color)
                else:
                    cv2.line(bg, start_xy, end_xy, (255,255,255), 1)
                if n < len(p_color):
                    transparency = 0.65
           
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    return img

def pkl_trajs_alphapose_trajs(trajs, show_input=False, vid='all'):
    """
    Run to put into vis_frame_format 
    """
    combine_trajs = []
    input_seq = trajs['id_y'].shape[-1]

    for elem in range(len(trajs['pred_trajs'])):
        if vid =='all':
            pass
        elif vid != trajs['video_file'][elem][:-4]:
            continue 

        flatten_trajs = []

        for seq_step in range(0,input_seq):
            poses_bbox= []
            pred_trajs = {}
            gt_trajs = {}
            
            if show_input:
                if trajs['pred_trajs'][elem].shape[-1] == 38:
                    gt_trajs['keypoints'] = trajs['x_ppl_box'][elem][seq_step,:-4].reshape(-1,2)
                    gt_trajs['box'] = trajs['x_ppl_box'][elem][seq_step,-4:]
                if trajs['pred_trajs'][elem].shape[-1] == 36:
                    gt_trajs['keypoints'] = trajs['x_ppl_box'][elem][seq_step,:].reshape(-1,2)

                gt_trajs['frame_x'] =  trajs['frame_x'][elem][seq_step]
                gt_trajs['idx'] = trajs['id_x'][elem][seq_step]
                gt_trajs['video_file'] = trajs['video_file'][elem][:-4]
                gt_trajs['abnormal_gt_frame'] = 0 # not lookin at abnormal inputs
                gt_trajs['abnormal_ped_input'] = trajs['abnormal_ped_input'][elem][seq_step]
                pred_trajs['abnormal_ped_pred'] = trajs['abnormal_ped_pred'][elem][seq_step]

                poses_bbox.append(gt_trajs)
            
            else:
                if trajs['pred_trajs'][elem].shape[-1] == 38:
                    pred_trajs['keypoints'] = trajs['pred_trajs'][elem][seq_step,:-4].reshape(-1,2)
                    pred_trajs['box'] = trajs['pred_trajs'][elem][seq_step,-4:]
                if trajs['pred_trajs'][elem].shape[-1] == 36:
                    pred_trajs['keypoints'] = trajs['pred_trajs'][elem][seq_step,:].reshape(-1,2)

                pred_trajs['frame_y'] = trajs['frame_y'][elem][seq_step]
                pred_trajs['idx'] = trajs['id_y'][elem][seq_step]
                pred_trajs['video_file'] = trajs['video_file'][elem][:-4]
                pred_trajs['abnormal_gt_frame'] = trajs['abnormal_gt_frame'][elem][seq_step]
                pred_trajs['abnormal_ped_pred'] = trajs['abnormal_ped_pred'][elem][seq_step]


                if trajs['pred_trajs'][elem].shape[-1] == 38:
                    gt_trajs['keypoints'] = trajs['y_ppl_box'][elem][seq_step,:-4].reshape(-1,2)
                    gt_trajs['box'] = trajs['y_ppl_box'][elem][seq_step,-4:]
                if trajs['pred_trajs'][elem].shape[-1] == 36:
                    gt_trajs['keypoints'] = trajs['y_ppl_box'][elem][seq_step,:].reshape(-1,2)
                
                gt_trajs['frame_y'] =  trajs['frame_y'][elem][seq_step]
                gt_trajs['idx'] = trajs['id_y'][elem][seq_step]
                gt_trajs['video_file'] = trajs['video_file'][elem][:-4]
                gt_trajs['abnormal_gt_frame'] = trajs['abnormal_gt_frame'][elem][seq_step]
                gt_trajs['abnormal_ped_pred'] = trajs['abnormal_ped_pred'][elem][seq_step]

                poses_bbox.append(pred_trajs)
                poses_bbox.append(gt_trajs)
            
            flatten_trajs.append( {'imgname': trajs['frame_y'][elem][seq_step], 'result': poses_bbox } )
        
        combine_trajs.append(flatten_trajs)
        
        

    return combine_trajs
            
def plot_pose_traj_entire_dataset(combine_trajs, plotted_pose_traj_path, dataset='avenue', show_input=False, tag_abnormal=False):

    opt = Opt()
    opt.tracking = False
    
    for trajs in combine_trajs:
        
        if tag_abnormal:
            is_traj_abnormal = []
            for pedestrain in trajs:
                is_traj_abnormal.append(pedestrain['result'][0]['abnormal_ped_pred'])
            is_traj_abnormal = np.array(is_traj_abnormal)   
            is_traj_abnormal = np.any(is_traj_abnormal==1)
        else:
            is_traj_abnormal = False
        
        for pedestrian in trajs:
            video_num = pedestrian['result'][0]['video_file']
            if show_input:
                frame_num = pedestrian['result'][0]['frame_x']
            else:
                frame_num = pedestrian['result'][0]['frame_y']
                
            idx = pedestrian['result'][0]['idx']
            if dataset == 'avenue':
                frame_loc = '/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/frames_of_vid/test/{:02d}/{:04d}.jpg'.format(int(video_num), frame_num) 
            elif dataset == 'st':
                frame_loc = os.path.join(loc['data_load']['st']['pic_loc_test'], '{}/{:03d}.jpg'.format(video_num, frame_num) )


            # Create traj directory
            if is_traj_abnormal:
                path_list = [ 'vid_{}_sframe_{:02d}_id_{:02d}_abnorm'.format( video_num, trajs[0]['imgname'], idx ) ]
                if show_input:
                    path_list = [ 'vid_{}_eframe_{:02d}_id_{:02d}_abnorm'.format( video_num, trajs[-1]['imgname']+1, idx ) ]
            else:
                path_list = [ 'vid_{}_sframe_{:02d}_id_{:02d}'.format( video_num, trajs[0]['imgname'], idx ) ]
                if show_input:
                    path_list = [ 'vid_{}_eframe_{:02d}_id_{:02d}'.format( video_num, trajs[-1]['imgname']+1, idx ) ]
                    
            # loc = '/home/akanu/results_all_datasets/pose/avenue_whole_correct_anom_lab/video_{:02d}'.format(int(video_num))
            location = os.path.join(plotted_pose_traj_path, 'video_{:02d}'.format(int(video_num)) ) 
            make_dir(path_list, location )    
            
            orig_img = cv2.imread(frame_loc)    
            img = vis_frame(orig_img, pedestrian, opt)
            
            
            if is_traj_abnormal:
                path = os.path.join( location,'vid_{}_sframe_{:02d}_id_{:02d}_abnorm'.format( video_num, trajs[0]['imgname'], idx) )
                if show_input:
                    path = os.path.join( location,'vid_{}_eframe_{:02d}_id_{:02d}_abnorm'.format( video_num, trajs[-1]['imgname']+1, idx) )
            else:
                path = os.path.join( location,'vid_{}_sframe_{:02d}_id_{:02d}'.format( video_num, trajs[0]['imgname'], idx) )
                if show_input:
                    path = os.path.join( location,'vid_{}_eframe_{:02d}_id_{:02d}'.format( video_num, trajs[-1]['imgname']+1, idx) )
                
                
            cv2.imwrite(os.path.join( path, '{:02d}.jpg'.format(frame_num)), img) 
                
def  save_pose_trajs_imgs_to_vids(imgs_path, save_vid_path, frame_rate=1):
    
    videos = os.listdir(imgs_path)
    videos.sort()
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    for video_num in videos: 
        
        path_list = [video_num]
        make_dir(path_list, save_vid_path)
        save_videos = os.path.join(save_vid_path, video_num)
        
        video_num = os.path.join(imgs_path, video_num)

        traj_pedestrains = os.listdir(video_num)
        traj_pedestrains.sort()
        
        
        for traj_pedestrain in traj_pedestrains:
            save_video = os.path.join(save_videos, '{}.avi'.format(traj_pedestrain))
            
            traj_pedestrain = os.path.join(video_num, traj_pedestrain)
            
            seq_pedestrain = os.listdir(traj_pedestrain)
            seq_pedestrain.sort()
            frames_to_vid = []
            
            for current_frame in seq_pedestrain:
                current_frame = os.path.join(traj_pedestrain, current_frame)

                img = cv2.imread(current_frame)
                frames_to_vid.append(img)
                
            height, width, layers = img.shape
            size = (width, height)

            out = cv2.VideoWriter(save_video, fourcc, frame_rate, size)
            
            for frame in frames_to_vid:
                out.write(frame)
            out.release()
            

if __name__ == '__main__':

    opt = Opt()
    in_len = 13
    dataset ='st'
    # Load data
    # load = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/using_gt_pred_endpoint_incorrectly/gaussian_avenue_in_{}_out_{}_K_1_pose_hc_endpoint.pkl'.format(in_len, in_len)
    load = '/home/akanu/output_bitrap/st_unimodal_pose_hc/using_incorrect_endpoint/gaussian_st_in_{}_out_{}_K_1_pose_hc_endpoint.pkl'.format(in_len, in_len)
    datas = load_pkl(load, dataset)
    
    # vis_trajs = pkl_trajs_alphapose_trajs(datas)
    vis_trajs = pkl_trajs_alphapose_trajs(datas, show_input=False, vid='01_0027')

    plotted_pose_traj_path = '/home/akanu/results_all_datasets/pose_hc/06_06_st_in_{}_out_{}_pose_hc_endpoint'.format(in_len, in_len)
    plot_pose_traj_entire_dataset(vis_trajs, os.path.join(plotted_pose_traj_path, 'images'), dataset = dataset,show_input=False, tag_abnormal=True)
    
    save_pose_trajs_imgs_to_vids( imgs_path=os.path.join(plotted_pose_traj_path, 'images'),
                                  save_vid_path=os.path.join(plotted_pose_traj_path, 'videos')
                                )
    quit()


    res = ind_seq_dict(datas, 1,240,7, 'vid_name')
