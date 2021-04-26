import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import os
import pickle
# from collections import OrderedDict
# from tensorflow.python.ops import math_ops
# does change show

def Files_Load(train_file,test_file):
    """
    This file seperates the unique locations of each file and
    the has and return output of the locations and file text.

    train_file: locations of the training bounding boxes
    test_file : locations of the testing bounding boxes
    This just put folder directory in a list for train and test

    return: dict with 'files_train', 'files_test', 'train_txt', 'test_txt'
    dict has list of the files 
    examples '/home/akanu/dataset/dataset/Anomaly/Avenue_Dataset/bounding_box_tlbr/Txt_Data/Train_Box/16.txt'

    or '01.txt'


    Potential fixes to make: remove the number that is returned by dict
    01.txt etc seems unnecessary

    """
    box_train_txt = os.listdir(train_file)
    box_train_txt.sort()
    box_test_txt = os.listdir(test_file)
    box_test_txt.sort()

    loc_files_train, loc_files_test = [], []

    for txt in box_train_txt:
        loc_files_train.append(os.path.join(train_file, txt))
        # loc_files_train.append(train_file + txt)
    for txt in box_test_txt:
        loc_files_test.append(os.path.join(test_file, txt))
        # loc_files_test.append(test_file + txt)



    locations = {}
    locations['files_train'] = loc_files_train
    locations['files_test'] = loc_files_test
    locations['txt_train'] = box_train_txt
    locations['txt_test'] = box_test_txt

    return locations

# def load_pkl(loc_files ):
def load_pkl():
    
    data_loc = '/home/akanu/output_bitrap/avenue/gaussian_avenue_640_360.pkl'
    temp, datadict = {}, {}
    
    with open(data_loc, 'rb') as f:
        data = pickle.load(f)
    # outputs = { 'X_global': all_X_globals, 'video_file': all_video_files,'abnormal':  all_abnormal,
    #                         'id_x':all_id_x, 'id_y': all_id_y , 'frame_x':all_frame_x, 'frame_y':all_frame_y,
    #                         'pred_trajs': all_pred_trajs, 'gt_trajs':all_gt_trajs,'distributions':distribution}
    # # creates empty list
    for key in data.keys():
        if key == 'distributions':
            continue
        
        temp[key] = []
        datadict[key] = 0
    
    # adds data
    for key in temp.keys():
        data_keyed = data[key]

        for data_elem in data_keyed:
            if key == 'pred_trajs':
                temp[key].append(data_elem[0][0]) # picking one of the 20 right here
            elif key == 'video_file':
                temp[key].append('{:02d}.txt'.format(data_elem))
            else:
                temp[key].append(data_elem)

    # puts data in array
    for key in temp.keys():
        print(key)
        datadict[key] = np.array(temp[key])

    temp = None #clears temp array
    
    #  rename keys
    datadict['x_ppl_box'] = datadict.pop('X_global')
    datadict['y_ppl_box'] = datadict.pop('gt_trajs')

    # add frame_ppl_id key
    frame_ppl_id = []
    for i,j,k,l in zip(data['frame_x'], datadict['frame_y'], datadict['id_x'], datadict['id_y']):
        frame = np.append(i,j)
        ids = np.append(k,l)

        frame_ppl_id.append( np.column_stack((frame, ids)) )

    datadict['frame_ppl_id'] = np.array(frame_ppl_id)
    return datadict



def Boxes(loc_files, txt_names, time_steps,data_consecutive, pad ='pre', to_xywh = False ):
    """
    This file process the bounding box data and creates a numpy array that
    can be put into a tensor



    
    loc_files: List that contains that has text files save
    txt_names: Txt file names. For visualization process
    time_step: Sequence length input. Also known as frames looked at for input
    pad: inputs 'pre' or 'post'. supplments short data with ones in front
    to_xywh: if given file is in tlbr and want to convert to xywb

    return:
    x_person_box: Has bounding box locations
    y_person_box: Label for bounding box locations
    frame_person_id: Contains frame Number and person_Id of entire sequence,
                     Last element is prediction frame. For visulization process
    video_file: Points to video file used. This gives me the frame video used
                and also the person id.

    abnormal: tells you wheather the frame is abnormal or not

    Potential fixes:
        Come back and seperate frame video and person id to make cleaner
    """
    #intilization 
    x_ppl_box, y_ppl_box, frame_ppl_id, video_file, abnormal = [], [], [], [],[]  #Has bounding box locations inside
    frame_x, frame_y, id_x, id_y = [], [], [], []

    #For splitting process
    split_train_test = 0
    split = 0
    find_split = 0

    # Tells me how many in sequence was short.
    # Do I want to go back and count for train and test separately
    short_len = 0

#     datadict = OrderedDict()
    datadict = {}

    for loc, txt_name in zip(loc_files, txt_names):
        data = pd.read_csv(loc, ' ' )
        # Note that person_box is 1 behind ID
        max_person = data['Person_ID'].max()
        for num in range(1,max_person+1):
            temp_box = data[data['Person_ID'] == num ]['BB_tl_0	BB_tl_1	BB_br_0	BB_br_1'.split()].values
            if to_xywh:
                temp_box = temp_box.astype(float)
                temp_box[:,2] = np.abs(temp_box[:,2] - temp_box[:,0] )
                temp_box[:,3] = np.abs(temp_box[:,3] - temp_box[:,1])

                temp_box[:,0] = temp_box[:,0] + temp_box[:,2]/2
                temp_box[:,1] = temp_box[:,1] + temp_box[:,3]/2

            person_seq_len = len(temp_box)
            # To split the frame and id to seperate code this is where I would make
            # initial change


            temp_frame_id = data[data['Person_ID'] == num ]['Frame_Number Person_ID'.split()].values
            abnormal_frame_ped = data[data['Person_ID'] == num]['anomaly'].values
            if person_seq_len > time_steps:
                # checks that data is sequenced correctly
                fix = temp_frame_id[:,0]
                cons_check = np.argsort(fix) == np.arange(0, len(fix), 1)
                if not np.all(cons_check):
                    print('Data not ordered for correct indexing /n sort data')
                    quit()
                    
                for i in range(0, person_seq_len - time_steps):
                    temp_fr_person_id = temp_frame_id[i:(i+time_steps+1)]
                    

                    x_frames = temp_fr_person_id[:-1,0]
                    y_frame = temp_fr_person_id[-1,0]
                    x_and_y = temp_fr_person_id[:,0]

                    if data_consecutive:
                        # This ensures that data inputted is from consecutive frames 
                        check_seq = np.cumsum(x_and_y) == np.cumsum(np.arange(y_frame -time_steps, y_frame + 1, 1))
                        if not np.all(check_seq):
                            continue
                    
                    temp_person_box = temp_box[i:(i+time_steps)]
                   

                    x_ppl_box.append(temp_person_box)
                    y_ppl_box.append(temp_box[i+time_steps])

                    assert temp_person_box.shape == (time_steps,4)
                    assert temp_fr_person_id.shape  == (time_steps+1,2), print(temp_fr_person_id.shape)

                    frame_ppl_id.append(temp_fr_person_id)
                    frame_x.append(x_frames)
                    frame_y.append(temp_fr_person_id[-1,0])
                    id_x.append(temp_fr_person_id[:-1,1])
                    id_y.append(temp_fr_person_id[-1,1])


                    video_file.append(txt_name)
                    abnormal.append(abnormal_frame_ped[i+time_steps]) #Finds if predicted frame is abnormal

            elif person_seq_len == 1:
                # want it to skip loop
                continue
            # This would add noise to data
            elif person_seq_len <= time_steps:
                continue
            #     temp_person_box_unpad = temp_box
            #     temp_fr_person_id_unpad = temp_frame_id
            #     temp_person_box = pad_sequences(temp_person_box_unpad.T, maxlen = time_steps+1, padding = pad).T
            #     temp_fr_person_id = pad_sequences(temp_fr_person_id_unpad.T,  maxlen = time_steps+1, padding = pad).T
            #
            #     assert temp_person_box.shape == (time_steps+1,4)
            #     assert temp_fr_person_id.shape  == (time_steps+1,2)
            #
            #     x_ppl_box.append(temp_person_box[0:time_steps,:])
            #     y_ppl_box.append(temp_person_box[time_steps,:])
            #
            #     frame_ppl_id.append(temp_fr_person_id[0:time_steps+1,:])
            #
            #     video_file.append(txt_name)
            #     abnormal.append(abnormal_frame_ped[-1]) #Finds if predicted frame is abnormal

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
    datadict['frame_ppl_id'] = np.array(frame_ppl_id) # delete later not needed once otuer changes 
    datadict['video_file'] = np.array(video_file)
    datadict['abnormal'] = np.array(abnormal, dtype=np.int8)
    datadict['id_x'] = np.array(id_x)
    datadict['id_y'] = np.array(id_y)
    datadict['frame_x'] = np.array(frame_x)
    datadict['frame_y'] = np.array(frame_y)

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


def norm_train_max_min(data, max1, min1, undo_norm=False):

    """
    data_dict: data input in the form of a dict. input is of same structure as
                output of  Boxes function

    data: data that is not in dictornary format that needs to be unnormailized
    max1 : normalizing parameter
    min1: normalizing parameter
    undo_norm: unnormailize data boolean

    return: depends on undo_norm boolean:
            if undo_norm is true return unnormailized data
            if undo_norm is false normailize


            
    """


    if undo_norm:
        # If data comes in here it is not in a dictornay format
        data = data*(max1-min1) + min1
        return data

    else:
        # If data comes in here it should be in the same 
        # format as Boxes function output partially 
        xx = (data['x_ppl_box'] - min1)/(max1 - min1)
        yy = (data['y_ppl_box'] - min1)/(max1 - min1)
        return xx,yy
