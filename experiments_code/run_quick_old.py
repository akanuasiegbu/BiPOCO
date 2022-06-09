import numpy as np

import tensorflow as tf
import os, time
from os.path import join

from custom_functions.utils import make_dir, SaveTextFile, write_to_txt, SaveAucTxt, SaveAucTxtTogether
from config.config import hyparams, loc, exp
from config.max_min_class_global import Global_Max_Min
from custom_functions.anomaly_detection import frame_traj_model_auc
from custom_functions.load_data import load_pkl



def run_quick(save_txt_loc, window_not_one = False):
    """
    window: changes the window size
    """
    norm_max_min = Global_Max_Min()

    # change this to run diff configs
    in_lens = [3,5,13,25]
    out_lens = [3,5,13,25]
    errors_type = ['error_summed', 'error_flattened' ]
    metrics =['l2']
    ablations =['hc_bone','hc_endpoint','hc_joint', 'hc_bone_endpoint', 'hc_bone_joint', 'hc_endpoint_joint', 'hc_all']
    # ablations =['scale_2', 'scale_3', 'scale_4', 'scale_5']
    for in_len, out_len in zip(in_lens, out_lens):
        for ablation in ablations:
            hyparams['input_seq'] = in_len
            hyparams['pred_seq'] = out_len
            # hyparams['input_seq'] = 13
            # hyparams['pred_seq'] = 13
            print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
            # continue
            if 'st' in exp['data'] and exp['model_name']=='bitrap':
                if window_not_one:
                    pklfile = loc['pkl_file']['st_template_skip'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'], hyparams['input_seq'])
                else:
                    pklfile = loc['pkl_file']['st_template'].format(hyparams['input_seq'], hyparams['pred_seq'], exp['K'])

            elif 'avenue' in exp['data'] and exp['model_name']=='bitrap':
                if window_not_one:
                    pklfile = loc['pkl_file']['avenue_template_skip'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'], hyparams['input_seq'])
                    print('I am here window not one')
                else:
                    # pklfile = loc['pkl_file']['avenue_template'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'])
                    pklfile = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/using_gt_pred_endpoint_incorrectly/gaussian_avenue_in_{}_out_{}_K_1_pose_{}.pkl'.format(hyparams['input_seq'],
                                                                                                                                    hyparams['pred_seq'], ablation )

            elif 'avenue' in exp['data'] and exp['model_name'] == 'lstm_network':
                if in_len in [3,13,25]:
                    modelpath = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
                else:
                    modelpath = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/05_18_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
                
            elif 'st' in exp['data'] and exp['model_name']== 'lstm_network':
                modelpath = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_st_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])

            if exp['model_name'] == 'lstm_network':
                model = tf.keras.models.load_model(     modelpath,  
                                                        custom_objects = {'loss':'mse'}, 
                                                        compile=True
                                                        )

                traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                                    loc['data_load'][exp['data']]['test_file'],
                                                    hyparams['input_seq'], hyparams['pred_seq'] 
                                                    )
                ## TO DO add norm_max_min for lstm!!!
                if window_not_one:
                    # Changes the window to run
                    traindict, testdict = data_lstm(    loc['data_load'][exp['data']]['train_file'],
                                                        loc['data_load'][exp['data']]['test_file'],
                                                        hyparams['input_seq'], hyparams['pred_seq'],
                                                        window = hyparams['input_seq']
                                                        )

            
            elif exp['model_name'] == 'bitrap':
                print(pklfile)                                                                                
                pkldicts = load_pkl(pklfile, exp['data'])
                model = 'bitrap'
            
            # for error in  ['error_diff', 'error_summed', 'error_flattened']:
            for error in errors_type:
                hyparams['errortype'] = error
                auc_metrics_list = []
                print(hyparams['errortype'])
                for metric in metrics:
                    hyparams['metric'] = metric
                    print(hyparams['metric'])
                    if exp['model_name'] == 'bitrap':
                        auc_metrics_list.append(frame_traj_model_auc( 'bitrap', [pkldicts], hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], norm_max_min))
                    elif exp['model_name'] == 'lstm_network':
                        auc_metrics_list.append(frame_traj_model_auc( [model], [testdict], hyparams['metric'], hyparams['avg_or_max'], exp['model_name'], norm_max_min))
                
                # path_list = loc['metrics_path_list'].copy()
                # path_list.append('{}_{}_in_{}_out_{}_K_{}'.format(loc['nc']['date'], exp['data'], hyparams['input_seq'],
                #                                                     hyparams['pred_seq'],exp['K'] ))
                # joint_txt_file_loc = join( os.path.dirname(os.getcwd()), *path_list )
                

                print(save_txt_loc)
                auc_together=np.array(auc_metrics_list)


                auc_slash_format = SaveAucTxtTogether(save_txt_loc)
                auc_slash_format.save(ablation, auc_together)


if __name__ == '__main__':
    save_txt_loc = '/home/akanu/results_all_datasets/pose_hc/{}_results_correct_endpoint'.format(exp['data'])
    # joint_txt_file_loc = join( save_txt_loc, '{}_{}_in_{}_out_{}_K_{}'.format(loc['nc']['date'], exp['data'], hyparams['input_seq'],
    #         hyparams['pred_seq'],exp['K'] ) )


    run_quick(save_txt_loc, window_not_one = False)