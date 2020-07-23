import pandas as pd
import cv2
# %matplotlib inline
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
# from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as kb
# from kerastuner.tuners import RandomSearch
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras.utils import normalize

from tensorflow_addons.utils.ensure_tf_install import _check_tf_version
from tensorflow_addons import losses


def data():

    def Files_Load():
        train_file = "/home/akanu/Dataset/Anomaly/Avenue_Dataset/bounding_box_tlbr/Txt_Data/Train_Box/"
        test_file = "/home/akanu/Dataset/Anomaly/Avenue_Dataset/bounding_box_tlbr/Txt_Data/Test_Box/"
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

        np.random.seed(49)
        rand = np.random.permutation(len(x_ppl_box))

        datadict['x_ppl_box'] = np.array(x_ppl_box)[rand]
        datadict['y_ppl_box'] = np.array(y_ppl_box)[rand]
        datadict['frame_ppl_id'] = np.array(frame_ppl_id)[rand]
        datadict['video_file'] = np.array(video_file)[rand]
        datadict['abnormal'] = np.array(abnormal)[rand]


        return  datadict



    len_frames =20

    loc_files_train, loc_files_test, box_train_txt, box_test_txt = Files_Load()
    traindict = Boxes(loc_files_train, box_train_txt, len_frames, pad ='pre')
    testdict = Boxes(loc_files_test[0:1], box_test_txt[0:1], len_frames, pad ='pre')


    max1 = traindict['x_ppl_box'].max()
    min1 = traindict['x_ppl_box'].min()
    x_train = (traindict['x_ppl_box'] - min1)/(max1 - min1)
    y_train = (traindict['y_ppl_box'] - min1)/(max1-min1)



    test_x = testdict['x_ppl_box']
    test_y = testdict['y_ppl_box']
    x_test = (test_x - min1)/(max1-min1)
    y_test = (test_y - min1)/(max1-min1)


    return x_train,y_train,x_test,y_test

def model1(x_train,y_train, x_test,y_test):
    def bb_intersection_over_union(y, x):
        xA = kb.max((x[:,0:1],y[:,0:1]), axis=0,keepdims=True)
        yA = kb.max((x[:,1:2],y[:,1:2]), axis=0,keepdims=True)
        xB = kb.min((x[:,2:3],y[:,2:3]), axis=0,keepdims=True)
        yB = kb.min((x[:,3:4],y[:,3:4]), axis=0,keepdims=True)

        interArea1 = kb.max((kb.zeros_like(xB), (xB-xA) ), axis=0, keepdims=True)
        interArea2 = kb.max((kb.zeros_like(xB), (yB-yA) ), axis=0, keepdims=True)
        interArea = interArea1*interArea2
        boxAArea = (x[:,2:3] - x[:,0:1] ) * (x[:,3:4] - x[:,1:2] )
        boxBArea = (y[:,2:3] - y[:,0:1] ) * (y[:,3:4] - y[:,1:2] )

        iou = interArea / (boxAArea + boxBArea - interArea)
        iou_mean = -kb.mean(iou)
        return iou_mean

    with tf.device('/GPU:0'):

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=x_train.shape[-2:]))

        model.add(keras.layers.LSTM({{choice([2,4,8,16,32,64,128,256])}},return_sequences =True ) )
        model.add(keras.layers.LSTM({{choice([2,4,8,16,32,64,128,256])}},return_sequences =True ) )
        model.add(keras.layers.LSTM({{choice([2,4,8,16,32,64,128,256])}},return_sequences =True ) )
        model.add(keras.layers.LSTM({{choice([2,4,8,16,32,64,128,256])}},return_sequences =True ) )
        model.add(keras.layers.LSTM({{choice([2,4,8,16,32,64,128,256])}},return_sequences =True ) )


        model.add(keras.layers.LSTM(4) )
        model.add(keras.layers.Dense(4) )

        opt = tf.keras.optimizers.Adam(learning_rate={{uniform(1e-7,1e-5)}})
        model.compile(loss=losses.GIoULoss(), optimizer=opt)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5)
        result = model.fit(x_train,y_train,
                          batch_size= 32,
                          epochs = 100,
                          validation_split=0.1,
                          callbacks =[early_stopping])

        validat_iou = np.amax(result.history['val_loss'])
        print('Best validation iou of epoch:', validat_iou)
    return {'loss': validat_iou, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    # with tf.device('/GPU:1'):
    best_run, best_model = optim.minimize(model=model1,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials(),
    				                      eval_space = True	)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
