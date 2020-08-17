import pandas as pd
import cv2
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
from load_data import Files_Load, Boxes, test_split_norm_abnorm, norm_train_max_min



def getDict(frames =20, startvid=0,endvid=1 ):
    loc_files_train, loc_files_test, box_train_txt, box_test_txt = Files_Load()
    traindict = Boxes(loc_files_train, box_train_txt, frames, pad ='pre')
    testdict = Boxes(loc_files_test[startvid:endvid], box_test_txt[startvid:endvid], frames, pad ='pre')
    return traindict,testdict



# Loading Data
train_file = "/home/akanu/Dataset/Anomaly/Avenue_Dataset/bounding_box_tlbr/Txt_Data/Train_Box/"
test_file = "/home/akanu/Dataset/Anomaly/Avenue_Dataset/bounding_box_tlbr/Txt_Data/Test_Box/"

frames = 20
startvid=0
endvid=1
loc_files_train, loc_files_test, box_train_txt, box_test_txt = Files_Load(train_file, test_file)
traindict = Boxes(loc_files_train, box_train_txt, frames, pad ='pre')
testdict = Boxes(loc_files_test[startvid:endvid], box_test_txt[startvid:endvid], frames, pad ='pre')
max1 = traindict['x_ppl_box'].max()
min1 = traindict['x_ppl_box'].min()
abnormal_dict, normal_dict = test_split_norm_abnorm(testdict)

# Normilize data
xx,yy = norm_train_max_min(data_dict = traindict, max1=max1,min1=min1)
xx_norm,yy_norm = norm_train_max_min(data_dict = normal_dict, max1=max1,min1=min1)
xx_abnorm,yy_abnorm = norm_train_max_min(data_dict = abnormal_dict, max1=max1,min1=min1)


print(xx[0:2])
