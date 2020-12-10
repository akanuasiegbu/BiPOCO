from load_data import Files_Load, Boxes
from pedsort import pedsort


from sklearn.model_selection import train_test_split

from config import hyparams
import tensorflow as tf
import numpy as np
def data_lstm(train_file, test_file):

    # returns a dict file
    loc = Files_Load(train_file,test_file)
    traindict = Boxes(  loc['files_train'], 
                        loc['txt_train'],
                        hyparams['frames'],
                        'pre', 
                        hyparams['to_xywh'] 
                        )

    testdict = Boxes(   loc['files_test'], 
                        loc['txt_test'],
                        hyparams['frames'], 
                        'pre',
                        hyparams['to_xywh']
                        )

    return traindict, testdict

def tensorify(train, val):

    """
    Mainly using this function to make training and validation sets
    train: dict that contains x and y
    val:dict that contains x and y

    return
    train_univariate: training tensor set
    val_univariate: validation tensor set
    """

    buffer_size = hyparams['buffer_size']
    batch_size = hyparams['batch_size']


    # print('Inside Tensorfify')
    """
    Come back and fix, I dont like tha I need to cast to float32
    """

    print(train['x'].shape)
    print(train['y'].shape)
    print(val['x'].shape)
    print(val['y'].shape)
    train_univariate = tf.data.Dataset.from_tensor_slices((train['x'], np.array(train['y'], dtype=np.float32)))
    train_univariate = train_univariate.cache().shuffle(buffer_size).batch(batch_size)
    val_univariate = tf.data.Dataset.from_tensor_slices((val['x'],np.array(val['y'], dtype=np.float32)))
    val_univariate = val_univariate.cache().shuffle(buffer_size).batch(batch_size)

    return train_univariate, val_univariate
