import numpy as np

import os, time
from os.path import join
from custom_functions.utils import make_dir

# Is hyperparameters and saving files config file
from config.config import hyparams, loc, exp
from config.max_min_class_global import Global_Max_Min



# Plots
from matplotlib import pyplot as plt

# Data Info
from custom_functions.load_data import  load_pkl

from custom_functions.ped_sequence_plot import ind_seq_dict 

# from custom_functions.frames_to_videos_n_back import convert_spec_frames_to_vid

from custom_functions.anomaly_detection import frame_traj_model_auc



def gpu_check():
    """
    return: True if gpu amount is greater than 1
    """
    return len(tf.config.experimental.list_physical_devices('GPU')) > 0




def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]



def plot_frame_wise_scores(out_frame):
    vid_to_split = np.unique(out_frame['vid'])

    out = {}
    for vid in vid_to_split:
        vid_index = np.where(out_frame['vid'] == vid)[0]
        # frames = np.array(out_frame['frame'], dtype=int)
        frames = out_frame['frame']
        framesort = np.argsort(frames[vid_index].reshape(-1))
        out[vid] = {}
        for key in out_frame.keys():
            out[vid][key] = out_frame[key][vid_index][framesort]

    
    # for key in out.keys():
    for key in out.keys():
        fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        # abnorm = np.where(out[key]['abnormal_gt_frame_metric'] == 1)[0]
        # norm = np.where(out[key]['abnormal_gt_frame_metric'] == 0)[0]
        # ax.scatter(out[key]['frame'][abnorm], out[key]['prob'][abnorm], marker='.', color ='r')
        # ax.scatter(out[key]['frame'][norm], out[key]['prob'][norm], marker='.', color ='b')
        ax.plot(out[key]['frame'],out[key]['prob'])

        index = np.where(out[key]['abnormal_gt_frame_metric'] ==1)[0]
        index_range =list(find_ranges(index))
        start = []
        end = []

        for i in index_range:
            if len(i) == 2:
                start.append(out[key]['frame'][i[0]])
                end.append(out[key]['frame'][i[1]])
            else:
                temp = out[key]['frame'][i[0]]
                start.append(temp)
                end.append(temp)
        

        for s,e in zip(start,end):
            ax.axvspan(s,e, facecolor='r', alpha=0.5)
        # ax.axvspan(299, 306, facecolor='b', alpha=0.5)
        # ax.axvspan(422, 493, facecolor='b', alpha=0.5)
        # ax.axvspan(562, 604, facecolor='b', alpha=0.5)

        ax.set_xlabel('Frames')
        ax.set_ylabel('Anomaly Score' )
        fig.savefig('testing_{}.jpg'.format(key[:-4]))  


def plot_traj_gen_traj_vid(testdict, modeltype, vid, frame, ped_id, norm_max_min, model=None):
    """
    testdict: the dict
    model: model to look at
    frame: frame number of interest (note assuming that we see the final frame first and)
            then go back and plot traj (int)
    ped_id: pedestrain id (int)
    vid: video number ('01_0014') ('05')

    """


    ped_loc = loc['visual_trajectory_list'].copy()
    
    person_seq = ind_seq_dict(testdict, '{}'.format(vid), frame,  ped_id) # this is a slow search I would think  
    
    ped_loc[-1] =  '{}'.format(vid) + '_' + '{}'.format(frame)+ '_' + '{}'.format(ped_id)
    make_dir(ped_loc)
    visual_plot_loc = join(     os.path.dirname(os.getcwd()),
                        *ped_loc
                        )
    vid_loc_gen = loc['data_load'][exp['data']]['pic_loc_test'] + '/{}/'.format(vid)


 

    plot_sequence(  person_seq,
                    norm_max_min.max,
                    norm_max_min.min,
                    pic_locs = vid_loc_gen,
                    visual_plot_loc = visual_plot_loc,
                    xywh= True
                    )

    gen_vid('{}_{}_{}'.format(vid, frame, ped_id), visual_plot_loc)


   

    

def main():

    model_loc = join(   os.path.dirname(os.getcwd()),
                        *loc['model_path_list']
                        ) # create save link
    

    norm_max_min = Global_Max_Min()

    


    file_to_load = '/home/akanu/output_bitrap/gaussian_avenue_in_3_out_3_K_1_pose_hc_endpoint.pkl'
    testdict = [ load_pkl(file_to_load, exp['data'], False) ]
    
    print( file_to_load)
    
    auc_human_frame = frame_traj_model_auc( 'bitrap', testdict, hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], norm_max_min)

    
    print('Input Seq: {}, Output Seq: {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
    print('Metric: {}, avg_or_max: {}'.format(hyparams['metric'], hyparams['avg_or_max']))
    print('Use kp confidence: {}'.format(exp['use_kp_confidence']))
    print('auc_human_frame human:{} frame:{}'.format(auc_human_frame[0], auc_human_frame[1]))
    return 


    

    

if __name__ == '__main__':
    main()

    print('Done') 