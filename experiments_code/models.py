import os
import tensorflow as tf
from tensorflow import keras

from config import hyparams

# To Do List
## Import time and make  

def lstm_network(train_data, val_data, model_loc, nc,  epochs=300):
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
        lstm_20.add(keras.layers.InputLayer(
            input_shape=(hyparams['frames'], 4)))
        lstm_20.add(keras.layers.LSTM(4, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(3, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(6, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(4, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(4, return_sequences=True))
        lstm_20.add(keras.layers.LSTM(4))
        lstm_20.add(keras.layers.Dense(4))
        opt = keras.optimizers.Adam(learning_rate=hyparams['newtorks']['lstm']['lr'])
        checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_loc, '{}_{}_{}_{}.h5'.format(nc[0], nc[1], nc[2], nc[3])),
                                                        save_best_only=True)

        if hyparams['newtorks']['lstm']['early_stopping'] == True:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=hyparams['newtorks']['lstm']['min_delta'],
                patience=hyparams['newtorks']['lstm']['patience'])

            cb = [early_stopping, checkpoint_cb]
        else:
            cb = [checkpoint_cb]

    #     lstm_20.compile(optimizer=opt, loss=losses.GIoULoss(), metrics=bb_intersection_over_union)
        # if use iou metric need to conver to tlbr
        lstm_20.compile(optimizer=opt, loss=hyparams['newtorks']['lstm']['loss'])
        lstm_20_history = lstm_20.fit(train_data,
                                      validation_data=val_data,
                                      epochs=epochs,
                                      callbacks=cb)
        return lstm_20_history , lstm_20


def binary_network(train_bm, val_bm, model_loc, nc,
                   weighted_binary, output_bias,
                   epochs=300, save_model=True):
    """
    train_bm: train_data_tensor
    val_bm: validation_data tensor
    model_loc : location to save models too
    nc: naming convention. list that contains [model, type,dataset,seq, abnormal_split]
        model: Dense_5_Drop_2
        type: xywh, tlbr
        dataset: ped1,ped2,st, avenue
        seq: size of sequence, 20, 5 etc int of sequence
        abnormal_split: percentage of abnormal frames to put in test frame

    weighted_binary: weighted_binary loss function
    output_bias: if not set output bias is set to 0 later in function
        """

    # gpu = '/device:GPU:'+gpu
    with tf.device('/device:GPU:0'):

        neurons = hyparams['newtorks']['binary_classifier']['neurons']
        dropout_ratio = hyparams['newtorks']['binary_classifier']['dropout']
        lr = hyparams['newtorks']['binary_classifier']['lr']
        # create model
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)

        model = keras.Sequential()
        model.add(keras.layers.Dense(neurons, input_dim=1, activation='relu'))
        model.add(keras.layers.Dense(neurons, activation='relu'))
        model.add(keras.layers.Dropout(dropout_ratio))  # comment
        model.add(keras.layers.Dense(neurons, input_dim=1, activation='relu'))
        model.add(keras.layers.Dense(
            neurons, input_dim=1, activation='relu'))  # coment
        model.add(keras.layers.Dropout(dropout_ratio))  # comment
        model.add(keras.layers.Dense(
            1, bias_initializer=output_bias, activation='sigmoid'))
        if output_bias is None:
            model.layers[-1].bias.assign([0.0])

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        

        
        checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_loc, '{}_{}_{}_{}.h5'.format(nc[0], nc[1], nc[2], nc[3])),
                                                        save_best_only=True)

        model.compile(loss=weighted_binary,
                      optimizer=opt, metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor=hyparams['newtorks']['binary_classifier']['mointor'],
                        min_delta=hyparams['newtorks']['binary_classifier']['min_delta'],
                        patience=hyparams['newtorks']['binary_classifier']['patience'],
                        restore_best_weights=True)

        if save_model and hyparams['newtorks']['binary_classifier']['early_stopping']:
            callbacks = [early_stopping, checkpoint_cb]
        elif early_stop:
            callbacks = [early_stopping]
        else:
            callbacks = None

        bm_history = model.fit(train_bm,
                               validation_data=val_bm,
                               epochs=epochs,
                               callbacks=callbacks)

        return bm_history, model
