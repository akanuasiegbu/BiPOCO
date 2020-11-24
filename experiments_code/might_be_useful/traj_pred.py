import pandas as pd
import cv2
%matplotlib inline
from matplotlib import pyplot as plt
import matplotlib
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def Files_Load():
    train_file = "/home/akanu/git/deep_sort_yolov3/Txt_Data/Train_Box/"
    test_file = "/home/akanu/git/deep_sort_yolov3/Txt_Data/Test_Box/"
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


    datadict['x_ppl_box'] = np.array(x_ppl_box)
    datadict['y_ppl_box'] = np.array(y_ppl_box)
    datadict['frame_ppl_id'] = np.array(frame_ppl_id)
    datadict['video_file'] = np.array(video_file)
    datadict['abnormal'] = np.array(abnormal)


    return  datadict



def videoPlot(y_ppl_box,y_pred, frame_ppl_id, video_file, vid_type ='test'):
    """
    Function is plotting the ground truth and predicted

    y_ppl_box: Label for bounding box locations
    y_ppl_box_pred: Prediction from Model
    frame_person_id: Contains frame Number and person_Id of entire sequence,
                     Last element is prediction frame. For visulization process
    video_file: Points to video file used. For visulization process
    """
    # Need a way to save images
    file = {}
    file['train'] = '/home/akanu/Dataset/Anomaly/Avenue_Dataset/training_videos/'
    file['test'] = "/home/akanu/Dataset/Anomaly/Avenue_Dataset/testing_videos/"
    loc_videos = file[vid_type] + video_file[0][:2] + '.' + video_file[0][2:5]
    ###

    video_capture = cv2.VideoCapture(loc_videos)

    for i in range(-1, frame_ppl_id[0,-1,0] + 1):
        #Assune sequences are connected and don't skip cuz of occlusions
        print(i)
        ret, frame = video_capture.read()
        if i == frame_ppl_id[0,-1,0]:
            pred_frame = frame.copy()
            cv2.rectangle(frame, (int(y_ppl_box[0,0]), int(y_ppl_box[0,1])), (int(y_ppl_box[0,2]), int(y_ppl_box[0,3])),(255,255,255), 2)
            cv2.putText(frame, str(frame_ppl_id[0,-1,1]),(int(y_ppl_box[0,0]), int(y_ppl_box[0,1])),0, 5e-3 * 200, (0,255,0),2)

            cv2.rectangle(pred_frame, (int(y_pred[0,0]), int(y_pred[0,1])), (int(y_pred[0,2]), int(y_pred[0,3])),(255,255,0), 2)
            cv2.putText(pred_frame, str(frame_ppl_id[0,-1,1]),(int(y_pred[0,0]), int(y_pred[0,1])),0, 5e-3 * 200, (0,255,0),2)

            cv2.imwrite('/home/akanu/git/deep_sort_yolov3/Images_saved/pred.jpg', pred_frame)
            cv2.imwrite("/home/akanu/git/deep_sort_yolov3/Images_saved/gt.jpg", frame)

def iou_function():
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        print(np.array(boxA))
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

#         print('here')
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
    return bb_intersection_over_union

# Iou metric. Should probably scale back to normal. Might not matter if not scaled back to normal

class IouMetric(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iou_avg = self.add_weight("iou_avg", initializer="zeros")
        self.bb_intersection_over_union = iou_function()
        self.count = self.add_weight("count", initializer="zeros")
    def update_state(self, y_true,y_pred):
#         print(y_pred)
        iou_calc =  self.bb_intersection_over_union(y_true,y_pred)
        self.iou_avg.assign_add(iou_calc)
        print('here')
        self.count.assign_add(1)
    def result(self):
        return self.iou_avg

if __name__ == "__main__":
    loc_files_train, loc_files_test, box_train_txt, box_test_txt = Files_Load()
    traindict = Boxes(loc_files_train, box_train_txt, 20, pad ='pre')
    testdict = Boxes(loc_files_test[3:4], box_test_txt[3:4], 20, pad ='pre')

    abnormal_index = np.nonzero(testdict['abnormal'])
    normal_index = np.where(testdict['abnormal'] == 0)

    abnormal_test_x = testdict['x_ppl_box'][abnormal_index]
    abnormal_test_y = testdict['y_ppl_box'][abnormal_index]
    test_x = testdict['x_ppl_box'][normal_index]
    test_y = testdict['y_ppl_box'][normal_index]

    max1 = traindict['x_ppl_box'].max()
    min1 = traindict['x_ppl_box'].min()
    xx = (traindict['x_ppl_box'] - min1)/(max1 - min1)
    xx_test_abnorm = (abnormal_test_x - min1)/(max1-min1)
    xx_test = (test_x - min1)/(max1-min1)


    yy = (traindict['y_ppl_box'] - min1)/(max1-min1)
    yy_test_abnorm = (abnormal_test_y - min1)/(max1-min1)
    yy_test = (test_y - min1)/(max1-min1)


    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    xx_train, xx_val,yy_train,yy_val = train_test_split(xx,yy, test_size = 0.1)
    train_univariate = tf.data.Dataset.from_tensor_slices((xx_train,yy_train))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_univariate = tf.data.Dataset.from_tensor_slices((xx_val,yy_val))
    val_univariate = val_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


    learning_rate = 0.0000005
    dense_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=xx.shape[-2:]),
    keras.layers.Dense(4)
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dense_model.compile(loss="mse", optimizer=opt, metrics=[IouMetric()])
    # history1 = dense_model.fit(xx, yy,validation_split=.1,
    #                     epochs=15)
    dense_history = dense_model.fit(train_univariate, epochs=5, validation_data =val_univariate  )
        
