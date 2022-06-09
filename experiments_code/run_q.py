import numpy as np

import tensorflow as tf
import os, time
from os.path import join

from custom_functions.utils import make_dir, SaveTextFile, write_to_txt, SaveAucTxt, SaveAucTxtTogether
from config.config import hyparams, loc, exp
from config.max_min_class_global import Global_Max_Min
from custom_functions.anomaly_detection import frame_traj_model_auc
from custom_functions.load_data import load_pkl



def run_quick(pkl_loc, save_txt_loc, ablations,norm_kp, window_not_one = False):
    """
    window: changes the window size
    """
    norm_max_min = Global_Max_Min()

    # change this to run diff configs
    in_lens = [3,5,13, 25]
    out_lens = [ 3,5,13, 25]
    errors_type = ['error_summed', 'error_flattened' ]
    metrics =['l2']
    
    # ablations =['scale_2', 'scale_3', 'scale_4', 'scale_5']
    for in_len, out_len in zip(in_lens, out_lens):
        for ablation in ablations:
            hyparams['input_seq'] = in_len
            hyparams['pred_seq'] = out_len
            # hyparams['input_seq'] = 13
            # hyparams['pred_seq'] = 13
            print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
            pklfile = join(pkl_loc, 'gaussian_st_in_{}_out_{}_K_1_pose_{}.pkl'.format(hyparams['input_seq'],
                                                                                                        hyparams['pred_seq'],
                                                                                                        ablation))
            # pklfile = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/using_gt_pred_endpoint_incorrectly/gaussian_avenue_in_{}_out_{}_K_1_pose_{}.pkl'.format(hyparams['input_seq'],
            #                                                                                                                         hyparams['pred_seq'], ablation )


            
            if exp['model_name'] == 'bitrap':
                print(pklfile)                                                                                
                pkldicts = load_pkl(pklfile, exp['data'], norm_kp)
                model = 'bitrap'
            
            for error in errors_type:
                hyparams['errortype'] = error
                auc_metrics_list = []
                print(hyparams['errortype'])
                for metric in metrics:
                    hyparams['metric'] = metric
                    print(hyparams['metric'])
                    if exp['model_name'] == 'bitrap':
                        auc_metrics_list.append(frame_traj_model_auc( 'bitrap', [pkldicts], hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], norm_max_min))
                    

                print(save_txt_loc)
                auc_together=np.array(auc_metrics_list)


                auc_slash_format = SaveAucTxtTogether(save_txt_loc)
                auc_slash_format.save(ablation, auc_together)


if __name__ == '__main__':
    # pkl_loc = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/using_gt_pred_endpoint_incorrectly'
    # pkl_loc = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/using_new_endpoint'

    # pkl_loc = '/home/akanu/output_bitrap/st_unimodal_pose_hc/using_incorrect_endpoint'
    
    # ablations =['hc_endpoint', 'hc_bone_endpoint', 'hc_endpoint_joint', 'hc_all']

    # For no bone joint endpoint
    # pkl_loc = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc'
    save_txt_loc = '/home/akanu/results_all_datasets/pose_hc/results_no_bone_endpoint_joint_not_norm/{}'.format(exp['data'])
    ablations = ['hc_no_bone_endpoint_joint']

    pkl_loc = '/home/akanu/output_bitrap/st_unimodal_pose_hc/using_incorrect_endpoint'
    # ablations =['hc_bone','hc_endpoint','hc_joint', 'hc_bone_endpoint', 'hc_bone_joint', 'hc_endpoint_joint', 'hc_all']
    # save_txt_loc = '/home/akanu/results_all_datasets/pose_hc/results_with_frame_norm_confidence_not_norm/{}_results_incorrect_endpoint'.format(exp['data'])
    
    run_quick(pkl_loc, save_txt_loc, ablations, norm_kp=False, window_not_one = False)