import pandas as pd
import numpy as np
import os
import pickle


def load_pkl(data_loc, dataset_name, normalize_kp_confidence=True):
    
    temp, datadict = {}, {}
    
    with open(data_loc, 'rb') as f:
        data = pickle.load(f)

    for key in data.keys():
        if key == 'distributions':
            continue
        
        temp[key] = []
        datadict[key] = 0
    
    for key in temp.keys():
        data_keyed = data[key]

        for data_elem in data_keyed:
            if key == 'pred_trajs':
                # Might need to figure out a way to fix this here
                preds = []
                for pred in data_elem:
                    preds.append(pred[0])
                temp[key].append(np.array(preds).reshape(-1, preds[0].shape[0]) ) 
            elif key == 'video_file':
                if 'avenue' in dataset_name:
                    temp[key].append('{:02d}.txt'.format(data_elem))
                elif 'st' in dataset_name:
                    elem = str(data_elem)

                    try:
                        vidnum = elem.split()
                  
                        if len(vidnum) == 3:
                            temp[key].append('{:02d}_{:04d}.txt'.format(int(vidnum[1]), int(vidnum[2][:-1])))
                        elif len(vidnum) == 2:
                            temp[key].append('{:02d}_{:04d}.txt'.format(int(vidnum[0][1:]), int(vidnum[1][:-1])))
                        else:
                            raise Exception('Other format!')
                    except:
                        if len(elem) == 5:
                            temp[key].append('0{}_{}.txt'.format(elem[0], elem[1:]))
                        elif len(elem) == 6:
                            temp[key].append('{}_{}.txt'.format(elem[0:2], elem[2:]))
                        else:
                            print('Error in load_pkl')
                            quit()
                elif 'corridor' in dataset_name:
                    temp[key].append('{:06d}.txt'.format(data_elem))


            else:
                temp[key].append(data_elem)

    for key in temp.keys():
        datadict[key] = np.array(temp[key])

    temp = None 
    #  rename keys
    if normalize_kp_confidence:
        datadict['kp_confidence_pred'] = normalize_confidence_for_poses(datadict['kp_confidence_pred'])
    
    datadict['x_ppl_box'] = datadict.pop('X_global')
    datadict['y_ppl_box'] = datadict.pop('gt_trajs')#.reshape(-1,4)

    # add frame_ppl_id key
    frame_ppl_id = []
    for i,j,k,l in zip(data['frame_x'], datadict['frame_y'], datadict['id_x'], datadict['id_y']):
        frame = np.append(i,j)
        ids = np.append(k,l)

        frame_ppl_id.append( np.column_stack((frame, ids)) )

    datadict['frame_ppl_id'] = np.array(frame_ppl_id)

    if dataset_name == 'hr-avenue':
        datadict = convert_pkl_to_HR_Avenue(datadict)
    elif dataset_name == 'hr-st':
        datadict = convert_pkl_to_HR_shanghaitech(datadict)
    return datadict

def convert_pkl_to_HR_Avenue(dataset):
    skip_frames_vid1 = [np.arange(75,121,1), np.arange(390, 437,1), np.arange(864,911,1), np.arange(931,1001, 1)]
    skip_frames_vid1 = np.concatenate(skip_frames_vid1)

    skip_frames_vid2 = [np.arange(272,320,1), np.arange(723,764,1) ]
    skip_frames_vid2 = np.concatenate(skip_frames_vid2)

    skip_frames_vid3 = [np.arange(293,341,1)]
    skip_frames_vid3 = np.concatenate(skip_frames_vid3)

    skip_frames_vid6 = [np.arange(561,625,1), np.arange(814,1007,1)]
    skip_frames_vid6 = np.concatenate(skip_frames_vid6)

    skip_frames_vid16 =[np.arange(728, 740,1)]
    skip_frames_vid16 = np.concatenate(skip_frames_vid16)

    vid_num = [ np.ones(skip_frames_vid1.shape,dtype=int ),
                2*np.ones(skip_frames_vid2.shape,dtype=int ),
                3*np.ones(skip_frames_vid3.shape,dtype=int ),
                6*np.ones(skip_frames_vid6.shape,dtype=int ),
                16*np.ones(skip_frames_vid16.shape,dtype=int )]
    
    skip_frames = np.concatenate((skip_frames_vid1,skip_frames_vid2, skip_frames_vid3,skip_frames_vid6, skip_frames_vid16))
    vid_num = np.concatenate(vid_num)


    # creates hash table for faster search
    search_dict = {}
    for vid in range(1,22):
        search_dict[vid] = []

    for  index, vid_frame in enumerate(zip(dataset['video_file'], dataset['frame_y'])):
        # append index, first frame, last frame
        first_frame, last_frame = vid_frame[1][0], vid_frame[1][-1]
        search_dict[int(vid_frame[0][:-4])].append([index, first_frame, last_frame])
    
    for vid in range(1,22):
        search_dict[vid] = np.array(search_dict[vid])


    remove_traj = []
    for frame, vid in zip(skip_frames, vid_num):
        one_video = search_dict[vid]

        found_traj_in_0_to_1 = (frame - one_video[:,1])/(one_video[:,2] - one_video[:,1])

        keep_index = np.where((found_traj_in_0_to_1 >= 0) & (found_traj_in_0_to_1 <=1))[0]

        remove_traj.append(one_video[keep_index][:,0]) #first value is index

    remove_traj = np.concatenate(remove_traj)
    remove_traj = np.unique(remove_traj) 


    dataset_HR_avenue = {}

    for key in dataset:
        dataset_HR_avenue[key] = np.delete(dataset[key], remove_traj, axis=0)

    return dataset_HR_avenue





def convert_pkl_to_HR_shanghaitech(dataset):
    assert 'txt' in dataset['video_file'][-1]
    vid_to_remove = ['01_0130.txt', '01_0135.txt', '01_0136.txt', '06_0144.txt', '06_0145.txt', '12_0152.txt']
    
    remove_index = []
    for index, vid in enumerate(dataset['video_file']):
        if vid in vid_to_remove:
            remove_index.append(index)        

    dataset_HR_st = {}

    for key in dataset:
        dataset_HR_st[key] = np.delete(dataset[key], remove_index, axis=0)
    
    return dataset_HR_st


def normalize_confidence_for_poses(unormalized_kp_confidence):
    temp_for_debug_summed_confidence = np.sum(unormalized_kp_confidence, axis=2)
    summed_confidence = np.repeat(  np.expand_dims(temp_for_debug_summed_confidence, axis=2), unormalized_kp_confidence.shape[-1], axis=2) 
    normalized_kp_confidence = np.divide(unormalized_kp_confidence, summed_confidence)
    
    return normalized_kp_confidence


def test_split_norm_abnorm(testdict):
    abnormal_index = np.nonzero(testdict['abnormal_ped_pred'])
    normal_index = np.where(testdict['abnormal_ped_pred'] == 0)
    normal_dict = {}
    abnormal_dict = {}

    for key in testdict.keys():
        normal_dict[key] = testdict[key][normal_index]
        abnormal_dict[key] = testdict[key][abnormal_index]


    return abnormal_dict, normal_dict


if __name__ == '__main__':
    
    # load ='/home/akanu/output_bitrap/avenue_unimodal_pose_hc/using_gt_pred_endpoint_incorrectly/gaussian_avenue_in_3_out_3_K_1_pose_hc_endpoint.pkl'
    # load = '/home/akanu/output_bitrap/avenue_unimodal/gaussian_avenue_in_3_out_3_K_1.pkl'

    load = '/home/akanu/output_bitrap/st_unimodal_pose_hc/using_incorrect_endpoint/gaussian_st_in_5_out_5_K_1_pose_hc_all.pkl'
    # load = '/home/akanu/output_bitrap/st_unimodal/gaussian_st_in_3_out_3_K_1.pkl'
    res = load_pkl(load, 'st', True)
    convert_pkl_to_HR_shanghaitech(res)
    # convert_pkl_to_HR_Avenue(res)    
    print('done')

