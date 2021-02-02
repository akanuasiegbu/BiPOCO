import numpy 

def ind_seq(data, video, frame):
    """
    Takes the data and find the video and frame so that 
    sequence can be inputted and creates smaller dict of one 
    sequence

    data: Format is keys are different videos
    video: video of interest. useful if in TN, TP, FN, FP. Ny that
            I mean that only inputed then the next immediate key is
            video numbers. For generic data this wouldn't work.
            As I would need to find the videos. Probably go back and 
            delete this funciton later on
    return output: same format as x_ppl_box,y_ppl_box etx
    """

    output ={}
    data = data['{}.txt'.format(video)]
    frames = data['frame_ppl_id'][:,-1,0]
    found_index_of_frame = np.where(frames==frame)

    
    for key in data.keys()

        output[key] = data[key][found_index_of_frame]

    
    return output

    

def plot_sequence(model, one_ped_seq, max1, min1, vid_key,pic_loc, loc_videos, xywh=False):
    """
    This will plot the sequences of the of one pedestrain
    Not computially efficent if you want to plot lots of pedestrain
    model: lstm bm_model
    one_ped_seq: one pedestrain sequence: 'x_ppl_box', 'y_ppl_box',
    'frame_ppl_id', 'video_file', 'abnormal'. From ind_seq function
    max1:  scaling factor
    min1:  scaline factor
    loc_videos: this are videos that are used to make plots
    pic_loc: need to save to different location depending on confusion
                    type and/or can input generic location to allow for
                    plotting all videos at the same sort_TP_TN_FP_FN_by_vid_n_frame
    """

    x_input = data['x_ppl_box']
    # x_scal,y_scal = norm_train_max_min(data_dict = data, max1 = max1,min1 =min1)
    last_frame = data['frame_ppl_id'][-1,-2,0]

    

    next_frame_index, j, frame_count = 0, 0, 0

    loc_vid = os.path.join(loc_videos, vid_key[:-4]+ '.avi')
    video_capture = cv2.VideoCapture(loc_vid)

    # there could be information lost here
    if xywh:
        x_input[:,0]  =  x_input[:,0]  -  x_input[:,2]/2
        x_input[:,1]  =  x_input[:,1]  -  x_input[:,3]/2 # Now we are at tlwh
        x_input[:,2:] =  x_input[:,:2] +  x_input[:,2:]


    x_input = x_input.squeeze()
    frame_ppl = data['frame_ppl_id'].squeeze()


    for i in range(0, last_frame+1):
        ret, frame = video_capture.read()
        if i == frame_ppl[j,0 ]: #finds the frames
            while i == frame_ppl[j,0]:

                x_box = x_input[j]
                id1 = frame_ppl[j,1]

                input_frame = frame.copy()
                # Since camera is statiornay I can plot other bbox as well on same video
                # Input Data
                cv2.rectangle(gt_frame, (int(x_box[0]), int(x_box[1])), (int(x_box[2]), int(x_box[3])),(0,255,0), 2)
                # Need to change This
                vid_str_info = vid_key[:-4] + '___' + str(i) + '__' + str(id1)
                cv2.imwrite( os.path.join(pic_loc, vid_str_info + '_input.jpg'), both_frame)
                
                frame_count += 1
                next_frame_index += 1
                j = next_frame_index
                
                if j == size:
                    break