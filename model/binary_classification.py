import tensorflow as tf
from tensorflow import keras
import os


def Dense_5_Drop_2 (train_bm =None,val_bm=None, model_loc=None, nc=None,
                        weighted_binary=None, output_bias = None, epochs=None):
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


    with tf.device('/device:GPU:0'):
        # create model

        neurons = 30
        dropout_ratio = 0.3
        lr = 0.00001
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)

        model = keras.Sequential()
        model.add(keras.layers.Dense(neurons, input_dim=1, activation='relu'))
        model.add(keras.layers.Dense(neurons, activation='relu'))
        model.add(keras.layers.Dropout(dropout_ratio)) #comment
        model.add(keras.layers.Dense(neurons, input_dim=1, activation='relu'))
        model.add(keras.layers.Dense(neurons, input_dim=1, activation='relu')) # coment
        model.add(keras.layers.Dropout(dropout_ratio)) #comment
        model.add(keras.layers.Dense(1, bias_initializer=output_bias))
        if output_bias is None:
            model.layers[-1].bias.assign([0.0])

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        x= (os.path.join(model_loc,'{}_{}_{}_{}_{:.3f}.h5'.format(nc[0], nc[1], nc[2], nc[3],nc[4] )))
        checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_loc,'{}_{}_{}_{}_{:.3f}.h5'.format(nc[0], nc[1], nc[2], nc[3],nc[4] ) ),
                                                                       save_best_only = True)



        model.compile(loss=weighted_binary, optimizer= opt, metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          min_delta=0.0005,
                                                          patience=10,
                                                          restore_best_weights=True)
        bm_history = model.fit(train_bm,
                     validation_data=val_bm,
                     epochs=epochs,
                     callbacks = [early_stopping, checkpoint_cb])



        return bm_history,model
