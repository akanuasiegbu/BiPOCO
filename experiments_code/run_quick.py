from custom_functions.utils import make_dir, SaveTextFile, write_to_txt, SaveAucTxt, SaveAucTxtTogether


def run_quick(window_not_one = False):
    """
    window: changes the window size
    """
    
    global max1, min1

    max1 = None
    min1 = None
    # change this to run diff configs
    # in_lens = [3,5,13,25]
    # out_lens = [3, 5,13,25]
    in_lens = [13]
    out_lens = [13]
    errors_type = ['error_summed', 'error_flattened']
    metrics = ['iou', 'giou', 'l2']

    for in_len, out_len in zip(in_lens, out_lens):
        hyparams['input_seq'] = in_len
        hyparams['pred_seq'] = out_len
        print('{} {}'.format(hyparams['input_seq'], hyparams['pred_seq']))
        # continue
        if exp['data']=='st' and exp['model_name']=='bitrap':
            if window_not_one:
                pklfile = loc['pkl_file']['st_template_skip'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'], hyparams['input_seq'])
            else:
                pklfile = loc['pkl_file']['st_template'].format(hyparams['input_seq'], hyparams['pred_seq'], exp['K'])

        elif exp['data']=='avenue' and exp['model_name']=='bitrap':
            if window_not_one:
                pklfile = loc['pkl_file']['avenue_template_skip'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'], hyparams['input_seq'])
                print('I am here window not one')
            else:
                pklfile = loc['pkl_file']['avenue_template'].format(hyparams['input_seq'], hyparams['pred_seq'],exp['K'])

        elif exp['data']=='avenue' and exp['model_name']=='lstm_network':
            if in_len in [3,13,25]:
                modelpath = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/07_05_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
            else:
                modelpath = '/home/akanu/results_all_datasets/experiment_traj_model/saved_model_consecutive/05_18_2021_lstm_network_xywh_avenue_{}_{}.h5'.format(hyparams['input_seq'], hyparams['pred_seq'])
            
        elif exp['data']=='st' and exp['model_name']=='lstm_network':
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
            # This sets the max1 and min1
            max1 = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
            min1 = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()

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
                    auc_metrics_list.append(frame_traj_model_auc( 'bitrap', [pkldicts], hyparams['metric'], hyparams['avg_or_max'], exp['model_name']))
                elif exp['model_name'] == 'lstm_network':
                    auc_metrics_list.append(frame_traj_model_auc( [model], [testdict], hyparams['metric'], hyparams['avg_or_max'], exp['model_name']))
            
            path_list = loc['metrics_path_list'].copy()
            path_list.append('{}_{}_in_{}_out_{}_K_{}'.format(loc['nc']['date'], exp['data'], hyparams['input_seq'],
                                                                hyparams['pred_seq'],exp['K'] ))
            joint_txt_file_loc = join( os.path.dirname(os.getcwd()), *path_list )

            print(joint_txt_file_loc)
            auc_together=np.array(auc_metrics_list)


            auc_slash_format = SaveAucTxtTogether(joint_txt_file_loc)
            auc_slash_format.save(auc_together)

            # auc_slash_format = SaveAucTxt(joint_txt_file_loc)



