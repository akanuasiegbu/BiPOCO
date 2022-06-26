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
from custom_functions.visualizations_pose import anomaly_score_frame_with_abnormal_regions_highlighted


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
    
    auc_human_frame, out_frame  = frame_traj_model_auc( 'bitrap', testdict, hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], norm_max_min)

    

    anomaly_score_figure = '../experiments_code/anomaly_score_plots'
    anomaly_score_frame_with_abnormal_regions_highlighted(out_frame, anomaly_score_figure)

    print('Input Seq: {}, Output Seq: {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
    print('Metric: {}, avg_or_max: {}'.format(hyparams['metric'], hyparams['avg_or_max']))
    print('Use kp confidence: {}'.format(exp['use_kp_confidence']))
    print('auc_frame:{}'.format(auc_human_frame[1]))
    

    

if __name__ == '__main__':
    main()

    print('Done') 