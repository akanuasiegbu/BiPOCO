from load_data import norm_train_max_min
import numpy as np
import cv2
import os


def cycle_through_videos(model, data, max1, min1,pic_loc, loc_videos, xywh=False):
    """
    The goal of the function is to allow data to be inputted as dict,
    that contains all videos and data is able to be plotted in order.
    A error might occur its probably because of data keys

    model: lstm bm_model
    data: Should contain dictornay keys to different videos. Each video
           should have same dictornay keys  'x_ppl_box' 'y_ppl_box'
               'video_file' 'abnormal'
               Note 'frame_ppl_id':  has format ( examples, seq input +seq output, (frame, ppl_id))
               should have made frame_ppl_id less confusing
    max1:  scaling factor
    min1:  scaline factor
    loc_videos: this are videos that are used to make plots
    pic_loc: need to save to different location depending on confusion
                    type and/or can input generic location to allow for
                    plotting all videos at the same sort_TP_TN_FP_FN_by_vid_n_frame
    """


    # Need to save to a different location depening on confusion type
    frame_count = 0
    for vid_key in data.keys():
#             print(vid_key)
            # Need to specifc folder location inside of loop
            vid_data = data[vid_key]
            x_scal,y_scal = norm_train_max_min(data_dict = vid_data, max1 = max1,min1 =min1)
            ## Sorting
            size = len(vid_data['frame_ppl_id'])
            frame = []
            for i in range(0,size):
                #sort index by frames
                # I go to last element because I Combined
                # the x and y frame and person id into one matrix
                frame.append(vid_data['frame_ppl_id'][i,-1,0])
            frame = np.array(frame)
            sort_index = frame.argsort()
            ## Sorting

            x_scal,y_scal = x_scal[sort_index] ,y_scal[sort_index]
            frame_ppl = vid_data['frame_ppl_id'][sort_index]
            y_true = vid_data['y_ppl_box'][sort_index] # not normailized
            print(vid_key)
            print(frame_ppl.shape)
            print(frame_ppl[:20,-1,0])
#             print(frame_ppl[:20,-1,1])


#             frame_count += frame_ppl.shape[0]
#             print(frame_count)
            y_scal_pred = model.predict(x_scal)
            y_pred = norm_train_max_min(data=y_scal_pred, max1 = max1,min1 =min1,undo_norm=True)

            last_frame = frame_ppl[-1,-1, 0]

            next_frame_index, j = 0, 0

            loc_vid = os.path.join(loc_videos, vid_key[:-4]+ '.avi')
            video_capture = cv2.VideoCapture(loc_vid)

            # there could be information lost here
            if xywh:
                y_pred[:,0] = y_pred[:,0] - y_pred[:,2]/2
                y_pred[:,1] = y_pred[:,1] - y_pred[:,3]/2 # Now we are at tlwh
                y_pred[:,2:] = y_pred[:,:2] + y_pred[:,2:]

                y_true[:,0] = y_true[:,0] - y_true[:,2]/2
                y_true[:,1] = y_true[:,1] - y_true[:,3]/2 # Now we are at tlwh
                y_true[:,2:] = y_true[:,:2] + y_true[:,2:]


            for i in range(0, last_frame+1):
                ret, frame = video_capture.read()
                if i == frame_ppl[j, -1,0 ]: #finds the frames
                    while i == frame_ppl[j,-1,0]:

                        y_fr_act = y_true[j]
                        y_fr_pred = y_pred[j]
                        id1 = frame_ppl[j,-1,1]

                        gt_frame = frame.copy()
                        pred_frame = frame.copy()
                        both_frame = frame.copy()

                        # Ground Truth
                        cv2.rectangle(gt_frame, (int(y_fr_act[0]), int(y_fr_act[1])), (int(y_fr_act[2]), int(y_fr_act[3])),(0,255,0), 2)

                        # Predicted
                        cv2.rectangle(pred_frame, (int(y_fr_pred[0]), int(y_fr_pred[1])), (int(y_fr_pred[2]), int(y_fr_pred[3])),(0,255,255), 2)

                        # Combined frame
                        cv2.rectangle(both_frame, (int(y_fr_act[0]), int(y_fr_act[1])), (int(y_fr_act[2]), int(y_fr_act[3])),(0,255,0), 2)
                        cv2.rectangle(both_frame, (int(y_fr_pred[0]), int(y_fr_pred[1])), (int(y_fr_pred[2]), int(y_fr_pred[3])),(0,255,255), 2)


                        # Need to change This
                        vid_str_info = vid_key[:-4] + '___' + str(i) + '__' + str(id1)
                        # vid_str_info has video number, frame number, person_Id

                        cv2.imwrite( os.path.join(pic_loc, vid_str_info + '_gt.jpg'), gt_frame)
                        cv2.imwrite( os.path.join(pic_loc, vid_str_info + '_pred.jpg'), pred_frame)
                        cv2.imwrite( os.path.join(pic_loc, vid_str_info + '_both.jpg'), both_frame)


                        frame_count += 1
                        print('vid:{} index:{} frame: {}'.format(
                                                                vid_key,
                                                                j,
                                                                frame_ppl[j,-1,0]))

                        next_frame_index += 1
                        j = next_frame_index

                        if j == size:
                            break
