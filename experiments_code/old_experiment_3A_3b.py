# To-Do Make sure that old experiment code runs
# Main thing to do is import the right things and
# Figure out a better way to transfer the max1 and min1


def classifer_train(traindict, testdict, lstm_model):
    
    # Loads based on experiment 
    train, val, test = data_binary(traindict, testdict, lstm_model, max1, min1)

    print("pos:{}, neg:{}".format(pos,neg))
    neg, pos = np.bincount(train['y'])
    print(pos/neg)
    # quit()

    
    # Removing the index so I can pass into tensofify and not have two
    # functions that do similar things.
    train_no_index, val_no_index = {},{}
    val_no_index['x'] = val['x'][:,0]
    val_no_index['y'] = val['y']
 
    train_no_index['x'] = train['x'][:,0]
    train_no_index['y'] = train['y']


    train_tensor, val_tensor = tensorify(
                                            train_no_index,    # print("pos:{}, neg:{}".format(pos,neg))

                                            val_no_index,
                                            batch_size = hyparams['networks']['binary_classifier']['batch_size'])

    #naming convection
    nc = [  loc['nc']['date'],
            loc['nc']['model_name_binary_classifer'],
            loc['nc']['data_coordinate_out'],
            loc['nc']['dataset_name'],
            hyparams['frames']
            ] # Note that frames is the sequence input


    make_dir(loc['model_path_list']) # Make directory to save model

    
    model_loc = join(   os.path.dirname(os.getcwd()),
                        *loc['model_path_list']
                        ) # create save link

    if not hyparams['networks']['binary_classifier']['wandb']:
        run = 'filler string'
    else:
        run = wandb.init(project="abnormal_pedestrain")    

    history, model= binary_network( train_tensor,
                                    val_tensor,
                                    model_loc=model_loc,
                                    nc =nc,
                                    weighted_binary=True, # should make clearer
                                    weight_ratio = pos/neg,
                                    output_bias =0, # Filler Play around with
                                    run = run, # this is for wandb
                                    epochs=hyparams['epochs'],
                                    save_model=hyparams['networks']['binary_classifier']['save_model'],
                                    )

    # test_auc_frame has iou on one column and index values on other column
    # remove_list contain pedestrain that are in the same frame as other
    # people in the test set for classifer

    
    # folders not saved by dates
    ###################################################################
    #  Training results
    ###################################################################
    make_dir(loc['metrics_path_list'])
    plot_loc = join(    os.path.dirname(os.getcwd()),
                        *loc['metrics_path_list']
                        )

    loss_plot(history, plot_loc, nc, save_wandb = True)
    accuracy_plot(history, plot_loc,nc)


    ###################################################################
    #  Testing results
    ###################################################################
    
    # Removes the index 
    test_no_index = {}
    test_no_index['x'] = test['x'][:,0]
    test_no_index['y'] = test['y']
    wandb_name = ['rocs', 'roc_curve']
    
    y_pred = model.predict(test_no_index['x'])
    y_true = test_no_index['y']
    roc_plot(y_true, y_pred, plot_loc, nc,wandb_name)
    
    # if exp['3_1']:
    test_auc_frame, removed_ped_index = ped_auc_to_frame_auc_data(model, testdict, test)
    # elif exp['3_2']:
    # For 3_2
    ########################################################################################################
    # Need to seperate test based on if their are negative index values
    # Note that negative index values means that selected test frame
    # came from the orgininal training frame in dictornary
        # from_traindict_index = np.where(test['x'][:,1] < 0)[0] # cuz of initializing the 0 of train as -0.01
        # from_testdict_index = np.where(test['x'][:,1] >= 0)[0] 
        
        # from_traindict  = {}
        # from_traindict['x'] = test['x'][from_traindict_index]
        # from_traindict['y'] = test['y'][from_traindict_index]
        # from_traindict_auc_frame, from_traindict_removed_ped_index = ped_auc_to_frame_auc_data(model, traindict, from_traindict)
        
        # from_testdict = {}
        # from_testdict['x'] = test['x'][from_testdict_index]
        # from_testdict['y'] = test['y'][from_testdict_index]
        # from_testdict_auc_frame, from_testdict_removed_ped_index = ped_auc_to_frame_auc_data(model, testdict, from_testdict)

        # test_auc_frame = {}
        # test_auc_frame['x'] = np.append(from_traindict_auc_frame['x'], from_testdict_auc_frame['x'], axis = 0)
        # test_auc_frame['y'] = np.append(from_traindict_auc_frame['y'], from_testdict_auc_frame['y'])
    ########################################################################################################



    # Removes the index 
    nc_frame = nc.copy()
    nc_frame[0] = loc['nc']['date'] + 'frame'
    test_no_index = {}
    test_no_index['x'] = test_auc_frame['x'][:,0]
    test_no_index['y'] = test_auc_frame['y']
    wandb_name_frame = ['rocs_frame', 'roc_curve_frame']
    print("this is nc_frame:{}".format(nc_frame))
    print("this is nc:{}".format(nc))

    y_pred = model.predict(test_no_index['x'])
    y_true = test_no_index['y']

    roc_plot(y_true, y_pred, plot_loc, nc_frame, wandb_name_frame)

    # should allow me to debug safety without abort program
    # run.finish()

    # quit()
    #########################################
    # More of a plotter than anything
    # Make into a function 
    # Looking at pedestrains that are deleted
    if len(removed_ped_index) >= 1:
        removed_ped = {}
        removed_ped['x'] = test['x'][removed_ped_index, :]
        removed_ped['y'] = test['y'][removed_ped_index]

        print('removed ped length x: {}'.format( len( removed_ped['x'] ) ) )
        print('removed ped length y: {}'.format( len( removed_ped['y'] ) ) )
        # seperates them into TP. TN, FP, FN
        conf_dict = seperate_misclassifed_examples( bm_model = model,
                                                    test_x = removed_ped['x'][:,0],
                                                    indices = removed_ped['x'][:,1],
                                                    test_y = removed_ped['y'],
                                                    threshold=0.5
                                                    )

        # print(len(conf_dict['TN']))
        # print(len(conf_dict['FN']))
        # print(len(conf_dict['FP']))
        # print(len(conf_dict['TP']))
        # print(conf_dict['TN'])
        # print(conf_dict['FN'])
        # print(conf_dict['FP'])
        # print(conf_dict['TP'])

        # quit()
        # what am I actually returning
        TP_TN_FP_FN, boxes_dict = sort_TP_TN_FP_FN_by_vid_n_frame(testdict, conf_dict )


        # Does not return result, but saves images to folders
        make_dir(loc['visual_trajectory_list'])
        pic_loc = join(     os.path.dirname(os.getcwd()),
                            *loc['visual_trajectory_list']
                            )

        # need to make last one robust "test_vid" : "train_vid"
        # can change

        loc_videos = loc['data_load'][exp['data']]['test_vid']
        # print(boxes_dict.keys())
        # quit()
        for conf_key in boxes_dict.keys():
            temp = loc['visual_trajectory_list'].copy()
            temp.append(conf_key)
            make_dir(temp)

        for conf_key in boxes_dict.keys():
            pic_loc_conf_key =  join(pic_loc, conf_key)
            cycle_through_videos(lstm_model, boxes_dict[conf_key], max1, min1, pic_loc_conf_key, loc_videos, xywh=True)

    print("pos:{}, neg:{}".format(pos,neg))

    # go back and fix filler
    print('go back and fix filler')
    
    # Need to save metric plots for classifer
