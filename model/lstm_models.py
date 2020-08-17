import tensorflow as tf
from tensorflow import keras
import os

def lstm_xywh_avenue_20(train_data = None, val_data=None ,model_loc = None, nc = None,  epochs =300):
    """
    train_data: train_data_tensor
    val_data: validation_data tensor
    model_loc : location to save models too
            nc: naming convention. list that contains [model, type,dataset,seq]
                model: lstm
                type: xywh, tlbr
                dataset: ped1,ped2,st, avenue
                seq: size of sequence, 20, 5 etc int of sequence
                example: lstm_xywh_ped1_20.h5
    """
    with tf.device('/device:GPU:0'):
        lstm_20 = keras.Sequential()
        # lstm_20.add(keras.layers.InputLayer(input_shape=xx.shape[-2:]))
        lstm_20.add(keras.layers.InputLayer(input_shape=(20,4)))
        lstm_20.add(keras.layers.LSTM(4,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(3,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(6,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(4,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(4,return_sequences =True ))
        lstm_20.add(keras.layers.LSTM(4) )
        lstm_20.add(keras.layers.Dense(4) )
        opt = keras.optimizers.Adam(learning_rate=8.726e-06)
        checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_loc,'{}_{}_{}_{}.h5'.format(nc[0], nc[1], nc[2], nc[3] ) ),
                                                                       save_best_only = True)

        early_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00005, patience=5)
    #     lstm_20.compile(optimizer=opt, loss=losses.GIoULoss(), metrics=bb_intersection_over_union)
        # if use iou metric need to conver to tlbr
        lstm_20.compile(optimizer=opt, loss='mse')
        lstm_20_history= lstm_20.fit(train_data,
                           validation_data = val_data,
                           epochs=epochs,
                           callbacks = [early_stopping, checkpoint_cb])
        return lstm_20_history
