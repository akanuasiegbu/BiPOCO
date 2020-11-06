import numpy as np


def seperate_misclassifed_examples(bm_model,test_x, test_y, threshold=0.5):
    """
    This function takes in the binary model and seperates out the index
    Values. Note that index values on the second row of test_x data.
    This index values allow for mapping back to test dataset.
    For an easy check compare numbers to confusion matrix with correct
    threshold.

    bm_model: binary classifer
    test_x: data we want to classify. These are iou points
    test_y: the ground truth of data
    return:
        index values seperate into FN and FP maybe a dictornay ??
    """
    # Now this seperate into True and False
    y_pred = bm_model.predict(test_x[0,:]) > threshold

    TN, TP, FP, FN = [], [], [], []
    index ={}

    # Key here is that the indices for the test data points
    # are on the second row
    for gt,pred,map_index in zip (test_y,y_pred,test_x[1,:]):
        if gt == False and pred == False:
            # This is for TN
            TN.append(map_index)
        elif gt == True and pred == False:
            # This one is FN
            FN.append(map_index)
        elif gt == False and pred == True:
            ## THis one is FP
            FP.append(map_index)
        else:
            # This one is for TP
            TP.append(map_index)

    index['TN'] = np.array([int(i) for i in TN])
    index['TP'] = np.array([int(i) for i in TP])
    index['FP'] = np.array([int(i) for i in FP])
    index['FN'] = np.array([int(i) for i in FN])

    return index


def sort_TP_TN_FP_FN_by_vid_n_frame(testdict, conf_dict ):
    """
    The goal of this function is seperate TP, TN, FP, FN indices
    by video and order each frame increasing to decreasing. Makes it
    easier to anlayze results as its more computially efficent to
    generate plots.

    testdict: this dict is in same format as Boxes function dict.
              dict contains 5 keys x_ppl_box, y_ppl_box, video_file,
              frame_ppl_id, abnormal

    conf_dict: contains indices that are correlated to the testdict.
                Indices are split based on TP, TN, FP, FN


    return:
        TP_TN_FP_FN: indices split up into specifc videos
        boxes_dict: testdict data parased and split up into specicfic videos
                    and into confusion matrix keys
    """

    TP_TN_FP_FN = {}
    boxes_dict = {}

    for conf_key in conf_dict.keys():
        # Need to seperate by video first
        # First line is the index of a specfic confusion matrix value
        # Those index map back to testdict unsorted
        unsorted_index_by_vid = conf_dict[conf_key]
    
        sorted_index = unsorted_index_by_vid.argsort()
        sorted_index_by_vid = unsorted_index_by_vid[sorted_index]

        sorted_video_list = testdict['video_file'][sorted_index_by_vid]

        prev = sorted_video_list[0]

        TP_TN_FP_FN[conf_key] = {}
        TP_TN_FP_FN[conf_key][prev] = []
        boxes_dict[conf_key] = {}

        #inital loop the prev is alawys equal to current
        # Then this next loop seperates into videos and ordered frames
        for current, j in zip(sorted_video_list,sorted_index_by_vid ):
            if prev != current:
                TP_TN_FP_FN[conf_key][current] = []
                TP_TN_FP_FN[conf_key][current].append(j)

            else:
                TP_TN_FP_FN[conf_key][current].append(j)

            prev = current

        # This look goes back and puts elements together
        for vid_key in TP_TN_FP_FN[conf_key].keys():
            boxes_dict[conf_key][vid_key] = {}
            index_per_vid= TP_TN_FP_FN[conf_key][vid_key]

            for attr_key in testdict.keys():
                temp = testdict[attr_key][ index_per_vid ]
                boxes_dict[conf_key][vid_key][attr_key] = temp

    return TP_TN_FP_FN,boxes_dict



    
