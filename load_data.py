import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import os
# from collections import OrderedDict
# from tensorflow.python.ops import math_ops


def Files_Load(train_file,test_file):

    box_train_txt = os.listdir(train_file)
    box_train_txt.sort()
    box_test_txt = os.listdir(test_file)
    box_test_txt.sort()

    loc_files_train, loc_files_test = [], []

    for txt in box_train_txt:
        loc_files_train.append(train_file + txt)
    for txt in box_test_txt:
        loc_files_test.append(test_file + txt)

    return loc_files_train, loc_files_test, box_train_txt, box_test_txt



def Boxes(loc_files, txt_names, time_steps, pad ='pre'):
    """
    loc_files: List that contains that has text files save
    txt_names: Txt file names. For visualization process
    time_step: Sequence length input
    pad: inputs 'pre' or 'post'

    x_person_box: Has bounding box locations
    y_person_box: Label for bounding box locations
    frame_person_id: Contains frame Number and person_Id of entire sequence,
                     Last element is prediction frame. For visulization process
    video_file: Points to video file used. For visulization process
    """

    x_ppl_box, y_ppl_box, frame_ppl_id, video_file, abnormal = [], [], [], [],[]  #Has bounding box locations inside

    #For splitting process
    split_train_test = 0
    split = 0
    find_split = 0

    # Tells me how many in sequence was short.
    # Do I want to go back and count for train and test seperatly
    short_len = 0

#     datadict = OrderedDict()
    datadict = {}

    for loc, txt_name in zip(loc_files, txt_names):
        data = pd.read_csv(loc, ' ' )
        # Note that person_box is 1 behind ID
        max_person = data['Person_ID'].max()
        for num in range(1,max_person+1):
            temp_box = data[data['Person_ID'] == num ]['BB_tl_0	BB_tl_1	BB_br_0	BB_br_1'.split()].values
            person_seq_len = len(temp_box)
            temp_frame_id = data[data['Person_ID'] == num ]['Frame_Number Person_ID'.split()].values
            abnormal_frame_ped = data[data['Person_ID'] == num]['anomaly'].values
            if person_seq_len > time_steps:
                for i in range(0, person_seq_len - time_steps):
                    temp_person_box = temp_box[i:(i+time_steps)]
                    temp_fr_person_id = temp_frame_id[i:(i+time_steps+1)]

                    x_ppl_box.append(temp_person_box)
                    y_ppl_box.append(temp_box[i+time_steps])

                    assert temp_person_box.shape == (time_steps,4)
                    assert temp_fr_person_id.shape  == (time_steps+1,2), print(temp_fr_person_id.shape)

                    frame_ppl_id.append(temp_fr_person_id)

                    video_file.append(txt_name)
                    abnormal.append(abnormal_frame_ped[i+time_steps]) #Finds if predicted frame is abnormal

            elif person_seq_len == 1:
                # want it to skip loop
                continue
            elif person_seq_len <= time_steps:
                temp_person_box_unpad = temp_box
                temp_fr_person_id_unpad = temp_frame_id
                temp_person_box = pad_sequences(temp_person_box_unpad.T, maxlen = time_steps+1, padding = pad).T
                temp_fr_person_id = pad_sequences(temp_fr_person_id_unpad.T,  maxlen = time_steps+1, padding = pad).T

                assert temp_person_box.shape == (time_steps+1,4)
                assert temp_fr_person_id.shape  == (time_steps+1,2)

                x_ppl_box.append(temp_person_box[0:time_steps,:])
                y_ppl_box.append(temp_person_box[time_steps,:])

                frame_ppl_id.append(temp_fr_person_id[0:time_steps+1,:])

                video_file.append(txt_name)
                abnormal.append(abnormal_frame_ped[-1]) #Finds if predicted frame is abnormal

            else:
                print('error')
#     np.random.seed(49)
#     rand = np.random.permutation(len(x_ppl_box))
#     datadict['x_ppl_box'] = np.array(x_ppl_box)[rand]
#     datadict['y_ppl_box'] = np.array(y_ppl_box)[rand]
#     datadict['frame_ppl_id'] = np.array(frame_ppl_id)[rand]
#     datadict['video_file'] = np.array(video_file)[rand]
#     datadict['abnormal'] = np.array(abnormal)[rand]

    datadict['x_ppl_box'] = np.array(x_ppl_box)
    datadict['y_ppl_box'] = np.array(y_ppl_box)
    datadict['frame_ppl_id'] = np.array(frame_ppl_id)
    datadict['video_file'] = np.array(video_file)
    datadict['abnormal'] = np.array(abnormal)

    return  datadict


def test_split_norm_abnorm(testdict):
    abnormal_index = np.nonzero(testdict['abnormal'])
    normal_index = np.where(testdict['abnormal'] == 0)
    normal_dict = {}
    abnormal_dict = {}

    for key in testdict.keys():
        normal_dict[key] = testdict[key][normal_index]
        abnormal_dict[key] = testdict[key][abnormal_index]


    return abnormal_dict, normal_dict


def norm_train_max_min(data_dict = None,data=None, max1=None, min1=None, undo_norm=False):

    if undo_norm:
        data = data*(max1-min1) + min1
        return data

    else:
        xx = (data_dict['x_ppl_box'] - min1)/(max1 - min1)
        yy = (data_dict['y_ppl_box'] - min1)/(max1-min1)
        return xx,yy
