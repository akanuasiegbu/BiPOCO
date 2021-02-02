import tensorflow
import os
import tensorflow as tf
from tensorflow import keras
from config import hyparams, exp

import numpy as np
from tensorflow.keras.layers import Lambda

def custom_loss(weight_ratio):
    """
    Note that weight_ratio is postive/negative
    """
    def loss(y_true, y_pred):
        when_y_1 = y_true*tf.keras.backend.log(y_pred)*(1/weight_ratio)
        # when_y_1 = y_true*tf.keras.backend.log(y_pred)*(1/1)
        neg_y_pred = Lambda(lambda x: -x)(y_pred)
        when_y_0 = ( 1+Lambda(lambda x: -x)(y_true))*tf.keras.backend.log(1+neg_y_pred )

        weighted_cross_entr = Lambda(lambda x: -x)(when_y_0+when_y_1)
        return weighted_cross_entr
    return loss


def main():

    weight_ratio =1/3
    abnorm_x = np.ones((500,2))
    norm_x = np.zeros((1500,2))
    data_x = np.append(abnorm_x, norm_x, 0)

    print(data_x.shape)

    abnorm_y = np.ones((500,1))
    norm_y = np.zeros((1500,1))
    data_y = np.append(abnorm_y, norm_y)
    print(data_y.shape)
    
    train_univariate = tf.data.Dataset.from_tensor_slices((data_x, data_y))
    train_univariate = train_univariate.cache().shuffle(96).batch(32)
    
    abnorm_x = np.ones((100,2))
    norm_x = np.zeros((300,2))
    data_x = np.append(abnorm_x, norm_x, 0)

    print(data_x.shape)

    abnorm_y = np.ones((100,1))
    norm_y = np.zeros((300,1))
    data_y = np.append(abnorm_y, norm_y)

    val_univariate = tf.data.Dataset.from_tensor_slices((data_x, data_y))
    val_univariate = val_univariate.cache().shuffle(96).batch(2)

    

    model = keras.Sequential()
    model.add(keras.layers.Dense(4, input_dim=2, activation='relu'))
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dropout(.3))  # comment
    model.add(keras.layers.Dense(4, input_dim=1, activation='relu'))
    model.add(keras.layers.Dense(
        4, input_dim=1, activation='relu'))  # coment
    model.add(keras.layers.Dropout(.3))  # comment
    model.add(keras.layers.Dense(
        1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=.001)
    
    model.compile(loss=custom_loss(weight_ratio),
                        optimizer=opt, metrics=['accuracy'])


    bm_history = model.fit( train_univariate,
                            validation_data=val_univariate,
                            epochs = 500
                            )           


if __name__ == '__main__':
    main()

    print('Done')