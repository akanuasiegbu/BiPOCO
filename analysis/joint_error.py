from custom_functions.anomaly_detection import ped_auc_to_frame_auc_data
from config.config import hyparams, loc, exp
from custom_functions.load_data import load_pkl
from config.max_min_class_global import Global_Max_Min
import numpy as np
from custom_functions.utils import SaveAucTxtTogether
from os.path import join
# for this to work note that hyparameter kp_confidence needs to be turned off

def joint_error(pkl_loc, save_joint_error_loc, ablations, norm_kp):
    exp['use_kp_confidence'] = False # should check that this always sets the default

    errors_type = ['error_summed']
    in_lens = [3,5,13, 25]
    out_lens = [ 3,5,13, 25]

    for in_len, out_len in zip(in_lens, out_lens):
        hyparams['input_seq'] = in_len
        hyparams['pred_seq'] = out_len
        save_to = SaveAucTxtTogether(save_joint_error_loc, joint_error=True)
        for ablation in ablations:
            file_to_load = join(pkl_loc, 'gaussian_st_in_{}_out_{}_K_1_pose_{}.pkl'.format(hyparams['input_seq'],
                                                                                                        hyparams['pred_seq'],
                                                                                                        ablation))
            # file_to_load = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/using_gt_pred_endpoint_incorrectly/gaussian_avenue_in_{}_out_{}_K_1_pose_{}.pkl'.format(hyparams['input_seq'],
            #                                                                                                                                 hyparams['pred_seq'],
            #                                                                                                                                 ablation )

            model = 'bitrap'
            testdicts = [ load_pkl(file_to_load, exp['data'], norm_kp) ]
            metric = hyparams['metric']
            print(metric)
            avg_or_max = hyparams['avg_or_max']
            modeltype = exp['model_name']
            norm_max_min = Global_Max_Min()

            for error in errors_type:
                hyparams['errortype'] = error
                print(hyparams['errortype'])

                test_auc_frame, remove_list, out_frame, out_human = ped_auc_to_frame_auc_data( model, testdicts, metric, avg_or_max, modeltype, norm_max_min, norm_scores_vid_wise = False)
                num_of_poses = out_frame['pred_bbox'].shape[-1]/2
                pred_len = hyparams['pred_seq']


                frame_joint_error_per_traj_in_pixels = out_frame['prob']/(num_of_poses*pred_len)
                human_joint_error_per_traj_in_pixels = out_human['prob']/(num_of_poses*pred_len)

                normal_index = np.where(out_frame['abnormal_gt_frame_metric'] == 0)[0]
                abnormal_index = np.where(out_frame['abnormal_gt_frame_metric'] == 1)[0]

                fn_mean = np.mean( frame_joint_error_per_traj_in_pixels[normal_index] )
                fab_mean = np.mean( frame_joint_error_per_traj_in_pixels[abnormal_index] )

                hn_mean = np.mean( human_joint_error_per_traj_in_pixels[normal_index] )
                hab_mean = np.mean( human_joint_error_per_traj_in_pixels[abnormal_index] )

                print(file_to_load)
                print('timescale {}'.format(hyparams['input_seq']))
                # print('{} {}'.format(fn_mean, fab_mean))
                save_to.save_joint_error(ablation, fn_mean,fab_mean, hn_mean, hab_mean)

if __name__ == '__main__':
    # pkl_loc = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/using_gt_pred_endpoint_incorrectly'
    # pkl_loc = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc/using_new_endpoint'

    # pkl_loc = '/home/akanu/output_bitrap/st_unimodal_pose_hc/using_incorrect_endpoint'
    # save_joint_error_loc = '/home/akanu/results_all_datasets/pose_hc/results_with_frame_norm_confidence_not_norm/{}_results_incorrect_endpoint'.format(exp['data'])

    # ablations =['hc_bone','hc_endpoint','hc_joint', 'hc_bone_endpoint', 'hc_bone_joint', 'hc_endpoint_joint', 'hc_all']
    # ablations =['hc_endpoint', 'hc_bone_endpoint', 'hc_endpoint_joint', 'hc_all']
    
    # #  for no bone endpoint joint
    pkl_loc = '/home/akanu/output_bitrap/st_unimodal_pose_hc/using_incorrect_endpoint'
    # pkl_loc = '/home/akanu/output_bitrap/avenue_unimodal_pose_hc'
    ablations = ['hc_no_bone_endpoint_joint']
    save_joint_error_loc = '/home/akanu/results_all_datasets/pose_hc/results_no_bone_endpoint_joint_not_norm/{}'.format(exp['data'])


    # pkl_loc = '/home/akanu/output_bitrap/st_unimodal_pose_hc/using_incorrect_endpoint'
    # ablations =['hc_bone','hc_endpoint','hc_joint', 'hc_bone_endpoint', 'hc_bone_joint', 'hc_endpoint_joint', 'hc_all']
    # save_joint_error_loc = '/home/akanu/results_all_datasets/pose_hc/results_with_frame_norm_confidence_not_norm/{}_results_incorrect_endpoint'.format(exp['data'])

    joint_error(pkl_loc, save_joint_error_loc, ablations, norm_kp=False)