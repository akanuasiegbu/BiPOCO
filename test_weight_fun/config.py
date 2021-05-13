import datetime

exp = { '1': True,
        '2': False,
        '3_1': False,
        '3_2': False,
        # want an adaptive model saved based on arch size for model_loc
        'data': 'avenue', #st, avenue,hr-st
        }

hyparams = {
    'epochs':30,
    'batch_size': 32,
    'buffer_size': 10000,
 
    'frames': 20,

    'to_xywh':True, # This is assuming file is in tlbr format
    # 'max':913.0, # wonder better way to pick
    # 'min':-138.5, # wonder better way to pick

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
            'neurons': 5,
            'dropout':0.3,
            'lr': 0.0001, #0.00001
            'save_model': False,
            'early_stopping':True,
            'mointor': 'loss',
            'min_delta':0.0000005,
            'batch_size': 32,
            'patience':30,
            'seed':40,
            'abnormal_split':0.5, # Split for the data into normal and abnormal
            'val_ratio':0.3, #guarantee the ratio of normal and abnormal frames
                            # are the same for validatin set and training set.
                            # so val_ratio. Think val_ratio(normal) + val_ratio(abnormal ) = val_ratio(normal + abnormal)

        }


    }

}

name_exp = None
if exp['1']:
    name_exp = '1'
elif exp['2']:
    name_exp = '2'
elif exp['3_1']:
    name_exp ='3_1'
elif exp['3_2']:
    name_exp = '3_2'

date = datetime.datetime.now()


loc =  {
    # if I'm running a test where don't want to save anything
    # how do I do that. Maybe move them to tmp
    
    'model_path_list': ['results_all_datasets', 'experiment_{}'.format(name_exp), 'saved_model'],
    'metrics_path_list': ['results_all_datasets', 'experiment_{}'.format(name_exp), 'metrics_plot'], 

    'nc':{
        'model_name': 'lstm_network',
        'model_name_binary_classifer': 'binary_network',
        'data_coordinate_out': 'xywh',
        'dataset_name': exp['data'], # avenue, st                   #################
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
            'hr-st':{
                'train_file':"/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/HR-ShanghaiTech/train_txt/",
                "test_file": "/mnt/roahm/users/akanu/projects/anomalous_pred/output_deepsort/HR-ShanghaiTech/test_txt/",
                },
            }


}