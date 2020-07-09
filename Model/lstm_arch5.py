import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time
import tensorflow.keras.backend as kb

def bb_intersection_over_union(y, x):
    xA = kb.max((x[:,0:1],y[:,0:1]), axis=0,keepdims=True)
    yA = kb.max((x[:,1:2],y[:,1:2]), axis=0,keepdims=True)
    xB = kb.min((x[:,2:3],y[:,2:3]), axis=0,keepdims=True)
    yB = kb.min((x[:,3:4],y[:,3:4]), axis=0,keepdims=True)

    interArea1 = kb.max((kb.zeros_like(xB), (xB-xA +1) ), axis=0, keepdims=True)
    interArea2 = kb.max((kb.zeros_like(xB), (yB-yA +1) ), axis=0, keepdims=True)
    interArea = interArea1*interArea2
    boxAArea = (x[:,2:3] - x[:,0:1] + 1) * (x[:,3:4] - x[:,1:2] + 1)
    boxBArea = (y[:,2:3] - y[:,0:1] + 1) * (y[:,3:4] - y[:,1:2] + 1)

    iou = interArea / (boxAArea + boxBArea - interArea)
    iou_mean = -kb.mean(iou)
    return iou_mean


def model(input):
    with tf.device('/device:GPU:1'):
        lstm_20 = keras.Sequential()
        lstm_20.add(keras.layers.InputLayer(input_shape=input.shape[-2:]))
        lstm_20.add(keras.layers.LSTM(4,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(3,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(6,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(4,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(4,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(4) )
        lstm_20.add(keras.layers.Dense(4) )
        opt = tf.keras.optimizers.Adam(learning_rate=8.726e-06)
        checkpoint_cb = keras.callbacks.ModelCheckpoint("lstm_5_arc_model.h5",
                                                           save_best_only = True)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5)
        lstm_20.compile(optimizer=opt, loss=bb_intersection_over_union, metrics='mse')

        lstm_20_history_1= lstm_20.fit(train_univariate,
                                   validation_data = val_univariate,
                                   epochs=100,
                                   callbacks = [early_stopping])

    return lstm_20
