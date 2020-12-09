hyparams = {
    'epochs':10,
    'batch_size': 32,
    'buffer_size': 10000,
    'val_size': 0.3,
    'learning_rate':0.001,

    'frames': 20,

    'to_xywh':True, # This is assuming file is in tlbr format
    'max':913.0, # wonder better way to pick
    'min':-138.5, # wonder better way to pick

    'newtorks': {
        'lstm':{
            'loss':'mse',
            'lr': 8.726e-06,
            'early_stopping': True,
            'mointor':'loss',
            'min_delta': 0.00005,
            'patience': 5
        },


        'binary_classifier':{
            'neurons': 30,
            'dropout':0.3,
            'lr': 0.00001,
            'early_stopping':True,
            'mointor': 'loss',
            'min_delta':0.00005,
            'patience':10,
            'seed':23,
            'abnormal_split':0.5, # Split for the data into normal and abnormal
            'val_ratio':0.3, #guarantee the ratio of normal and abnormal frames
                            # are the same for validatin set and training set.
                            # so val_ratio. Think val_ratio(normal) + val_ratio(abnormal ) = val_ratio(normal + abnormal)
                            
        }


    }

}


# want an adaptive model saved based on arch size for model_loc

loc =  {
    # if I'm running a test where don't want to save anything 
    # how do I do that. Maybe move them to tmp
    'model_path_list': ['results_all_datasets', 'experiment_1', 'avenue', 'saved_model'],
    'metrics_path_list': ['results_all_datasets', 'experiment_1', 'avenue', 'metrics_plot'],
    
    'nc':{  
        'model_name': 'lstm_network',
        'model_name_binary_classifer': 'binary_network', 
        'data_coordinate_out': 'xywh',
        'dataset_name': 'avenue' # avenue, st
        },
    # is nc the best way to propate and save things as same name 
    # Might want to autmoically create a new folder with model arch saved
    # as a text file as well as in folder name

    'data_load':{
            'avenue':{
                # These are good because these locations are perm unless I manually move them
                'train_file': "/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/bounding_box_tlbr/Txt_Data/Train_Box/",
                'test_file': "/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/bounding_box_tlbr/Txt_Data/Test_Box/",
                }
            }
}
