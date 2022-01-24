
import os
import numpy as np

from config.config import hyparams, loc, exp

from custom_functions.anomaly_detection import ped_auc_to_frame_auc_data
from custom_functions.load_data import load_pkl
from custom_functions.visualizations import plot_error_in_time
from custom_functions.utils import make_dir

if __name__ == '__main__':
    testdict = [ load_pkl('/home/akanu/output_bitrap/avenue_unimodal_pose/gaussian_avenue_in_{}_out_{}_K_1_pose.pkl'.format(hyparams['input_seq'],
                                                                                                                                hyparams['pred_seq']),
                                                                                                                                exp['data']) ]
    
    # don't need max1 or min1
    test_auc_frame, remove_list, out_frame, out_human = ped_auc_to_frame_auc_data('bitrap', testdict, hyparams['metric'], hyparams['avg_or_max'], exp['model_name'])
    
    time_series = ['results_all_datasets','pose', 'avenue_whole_prob_along_time_anom_lab'] 
    time_series_plot_loc = os.path.join( os.path.dirname(os.getcwd()) , *time_series )
    
    if exp['data'] =='avenue':
        for i in range(1,22):
            path = time_series.copy()
            path.append('{:02d}_time_series'.format(i))
            make_dir(path)
    
    for prob_with_time, vid, frame, idy, abnormal_ped_pred in zip(  out_human['prob_with_time'][7658:],
                                                                    out_human['vid'][7658:], 
                                                                    out_human['frame'][7658:], 
                                                                    out_human['id_y'][7658:],
                                                                    out_human['abnormal_ped_pred'][7658:]
                                                                    ):
    
        if not hyparams['errortype']=='error_flattened':
            is_traj_abnormal = np.any(abnormal_ped_pred == 1)
            plot_error_in_time(prob_with_time, time_series_plot_loc, vid[0][:-4], int(frame[0]), int(idy[0]), is_traj_abnormal)