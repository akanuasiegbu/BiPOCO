
import os
import numpy as np

from config.config import hyparams, loc, exp
from config.max_min_class_global import Global_Max_Min

from custom_functions.anomaly_detection import ped_auc_to_frame_auc_data
from custom_functions.load_data import load_pkl
from custom_functions.visualizations import plot_error_in_time
from custom_functions.utils import make_dir
from custom_functions.search import pkl_seq_ind

if __name__ == '__main__':
    testdict = [ load_pkl('/home/akanu/output_bitrap/avenue_unimodal_pose/gaussian_avenue_in_{}_out_{}_K_1_pose.pkl'.format(hyparams['input_seq'],
                                                                                                                                hyparams['pred_seq']),
                                                                                                                                exp['data']) ]
    
    # don't need max1 or min1
    norm_max_min = Global_Max_Min()
    test_auc_frame, remove_list, out_frame, out_human = ped_auc_to_frame_auc_data('bitrap', testdict, hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], norm_max_min)
    
    time_series = ['results_all_datasets','pose', 'avenue_in_{}_out_{}'.format(hyparams['input_seq'],
                                                                                hyparams['pred_seq']), 'avenue_whole_prob_along_time_anom_lab_without_conf'] 
    time_series_plot_loc = os.path.join( os.path.dirname(os.getcwd()) , *time_series )
    
    found = pkl_seq_ind(out_human,'48', '07', '16')
    found_1 = pkl_seq_ind(out_human,'399', '04', '25')
    found_1 = pkl_seq_ind(out_human,'39', '04', '04')
    
    if exp['data'] =='avenue':
        for i in range(1,22):
            path = time_series.copy()
            path.append('{:02d}_time_series'.format(i))
            make_dir(path)
    
    for prob_with_time, vid, frame, idy, abnormal_ped_pred in zip(  out_human['prob_with_time'][12200:15300],
                                                                    out_human['vid'][12200:15300], 
                                                                    out_human['frame'][12200:15300], 
                                                                    out_human['id_y'][12200:15300],
                                                                    out_human['abnormal_ped_pred'][12200:15300]
                                                                    ):
    
        if not hyparams['errortype']=='error_flattened':
            is_traj_abnormal = np.any(abnormal_ped_pred == 1)
            plot_error_in_time(prob_with_time, time_series_plot_loc, vid[0][:-4], int(frame[0]), int(idy[0]), is_traj_abnormal)