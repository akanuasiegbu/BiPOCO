import datetime
# change 3 things 

hyparams = {
    'epochs':300,
    'batch_size': 320, #32
    'buffer_size': 10000,
 
    'frames': 20,

    'to_xywh':True, # This is assuming file is in tlbr format
    # 'max':913.0, # wonder better way to pick
    # 'min':-138.5, # wonder better way to pick

    'exp_1': False, # might not be best place
    'exp_3': False, # might not be best place
    # ________________________________________________________________________________________________________________________________--
    # True False exp_1
    # False #False is exp_2
    # True True exp_31
    # False , True for exp 3_2

    'networks': {
        'lstm':{
            'loss':'mse',
            'lr': 8.726e-06,
            'early_stopping': True,
            'mointor':'loss',
            'min_delta': 0.00005,
            'patience': 15,
            'val_ratio': 0.3,
        },


        'binary_classifier':{
            'neurons': 30,
            'dropout':0.3,
            'lr': 0.0001, #0.00001
            'early_stopping':False,
            'mointor': 'loss',
            'min_delta':0.0000005,
            'patience':30,
            'seed':40,
            'abnormal_split':0.5, # Split for the data into normal and abnormal
            'val_ratio':0.3, #guarantee the ratio of normal and abnormal frames
                            # are the same for validatin set and training set.
                            # so val_ratio. Think val_ratio(normal) + val_ratio(abnormal ) = val_ratio(normal + abnormal)

        }


    }

}

date = datetime.datetime.now()
# want an adaptive model saved based on arch size for model_loc
# data = 'avenue'

loc =  {
    # if I'm running a test where don't want to save anything
    # how do I do that. Maybe move them to tmp
    
    # 'model_path_list': ['results_all_datasets', 'experiment_1', 'saved_model'],
    # 'metrics_path_list': ['results_all_datasets', 'experiment_1', 'metrics_plot'], 

    'model_path_list': ['results_all_datasets', 'experiment_2', 'saved_model'],
    'metrics_path_list': ['results_all_datasets', 'experiment_2', 'metrics_plot'], 

    # 'model_path_list': ['results_all_datasets', 'experiment_3_1', 'saved_model'],
    # 'metrics_path_list': ['results_all_datasets', 'experiment_3_1', 'metrics_plot'], 
    # # Intent to make easier to view results

    # 'model_path_list': ['results_all_datasets', 'experiment_3_2', 'saved_model'],
    # 'metrics_path_list': ['results_all_datasets', 'experiment_3_2', 'metrics_plot'], 

    'nc':{
        'model_name': 'lstm_network',
        'model_name_binary_classifer': 'binary_network_lr_0.0001_batch_320',
        'data_coordinate_out': 'xywh',
        'dataset_name': 'st', # avenue, st                   #################
        # ________________________________________________________________________________________________________________________________--
        'date': date.strftime("%m") + '_' +date.strftime("%d") + '_' + date.strftime("%Y"),
        },
    # is nc the best way to propate and save things as same name
    # Might want to autmoically create a new folder with model arch saved
    # as a text file as well as in folder name

    'data_load':{
            'avenue':{
                # These are good because these locations are perm unless I manually move them
                'train_file': "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/avenue/train_txt/",
                'test_file': "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/avenue/test_txt/",
                },

            #Need to rerun ped1
            # 'ped1':{
            #     'train_file':"/mnt/roahm/users/akanu/dataset/Anomaly/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Txt_Data/Train_Box_ped1/",
            #     "test_file": "/mnt/roahm/users/akanu/dataset/Anomaly/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Txt_Data/Test_Box_ped1/",
            #     },
            'st':{
                'train_file':"/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/st/train_txt/",
                "test_file": "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/st/test_txt/",
                },
            }
}
