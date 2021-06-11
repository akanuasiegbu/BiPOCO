import cv2
import numpy as np
from os.path import join


def plot_frame_from_image(pic_loc, bbox_preds, save_to_loc, vid, frame, idy, prob,abnormal_frame, abnormal_ped, gt_bbox = None):
    """
    pic_loc: this is where the orginal pic is saved
    bbox_pred: this is where the bbox is saved
    """
    img = cv2.imread(pic_loc)
    for bbox_pred in bbox_preds:
        cv2.rectangle(img, (int(bbox_pred[0]), int(bbox_pred[1])), (int(bbox_pred[2]), int(bbox_pred[3])),(0,255,255), 2)
    
    
    cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])),(0,255,0), 2)

    # cv2.putText(img, '{:.4f}'.format(prob),(int(bbox_pred[2]), int(bbox_pred[3])),0, 5e-3 * 100, (255,255,0),2)

    cv2.putText(img, '{:.4f}'.format(prob),(25,65),0, 5e-3 * 100, (255,255,0),2)
    
    # This is for abnormal frame, this uses the last bbox of set to plot 
    if abnormal_frame:
        # cv2.putText(img, '1',(int(bbox_pred[0]), int(bbox_pred[1])),0, 5e-3 * 100, (0,0,255),2)
        cv2.putText(img, 'Abnormal',(25,25),0, 5e-3 * 150, (0,0,255),2)
    else:
        # cv2.putText(img, '0',(int(bbox_pred[0]), int(bbox_pred[1])),0, 5e-3 * 100, (0,255, 0 ),2)
        cv2.putText(img, 'Normal',(25, 25),0, 5e-3 * 150, (0,255, 0 ),2)
    
    # This is for abnormal person
    if abnormal_ped:
        # cv2.putText(img, '1',(int(bbox_pred[0]), int(bbox_pred[1])),0, 5e-3 * 100, (0,0,255),2)
        cv2.putText(img, '1',(25,45),0, 5e-3 * 100, (0,0,255),2)

    else:
        # cv2.putText(img, '0',(int(bbox_pred[0]), int(bbox_pred[1])),0, 5e-3 * 100, (255,0, 0),2)
        cv2.putText(img, '0',(25,45),0, 5e-3 * 100, (255,0, 0),2)



    # if abnormal_frame and abnormal_ped:
    
    cv2.imwrite(save_to_loc + '/' + '{:02d}__{}_{}_{:.4f}.jpg'.format(vid, frame, idy, prob), img)


def plot_vid(out_frame, pic_loc, visual_plot_loc):

    """
    out_frame: this is the dict
    pic_loc: pic that will be plottd
    visual_plot_loc: this is where the video will be saved at
    """
    for bbox_preds, vid, frame, idy, prob, abnormal_frame, abnormal_ped, gt_bbox in zip(    out_frame['bbox'], out_frame['vid'], 
                                                                                            out_frame['frame'], out_frame['id_y'],
                                                                                            out_frame['prob'],
                                                                                            out_frame['abnormal_gt_frame_metric'],
                                                                                            out_frame['abnormal_ped_pred'],
                                                                                            out_frame['gt_bbox'] ):
        pic_loc = join(  pic_loc, '{:02d}'.format(int(vid[0][:-4])) )
        pic_loc =  pic_loc + '/' +'{:02d}.jpg'.format(int(frame))
        plot_frame_from_image(  pic_loc = pic_loc,  
                                bbox_preds = bbox_preds ,
                                save_to_loc = visual_plot_loc, 
                                vid = int(vid[0][:-4]),
                                frame = int(frame[0]), 
                                idy = int(idy[0]),
                                prob = prob[0],
                                abnormal_frame = abnormal_frame,
                                abnormal_ped = abnormal_ped,
                                gt_bbox = gt_bbox)