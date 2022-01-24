# This is the higher level data loading
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split



from custom_functions.load_data import Files_Load, Boxes
from custom_functions.load_data import norm_train_max_min

from config.config import hyparams, exp

def data_lstm(train_file, test_file, input_seq, pred_seq, window=1):

    # returns a dict file
    loc = Files_Load(train_file,test_file)
    traindict = Boxes(  loc_files = loc['files_train'], 
                        txt_names = loc['txt_train'],
                        input_seq = input_seq,
                        pred_seq = pred_seq,
                        data_consecutive = exp['data_consecutive'], 
                        pad = 'pre', 
                        to_xywh = hyparams['to_xywh'],
                        testing = False,
                        window = window
                        )

    testdict = Boxes(   loc_files = loc['files_test'], 
                        txt_names = loc['txt_test'],
                        input_seq = hyparams['input_seq'],
                        pred_seq = hyparams['pred_seq'], 
                        data_consecutive = exp['data_consecutive'],
                        pad = 'pre',
                        to_xywh = hyparams['to_xywh'],
                        testing = True,
                        window = window 
                        )
                        
    return traindict, testdict

def tensorify(train, val, batch_size):

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

    train_univariate = tf.data.Dataset.from_tensor_slices((train['x'], np.array(train['y'].reshape(-1,hyparams['pred_seq']*4), dtype=np.float32)))
    train_univariate = train_univariate.cache().shuffle(buffer_size).batch(batch_size)
    val_univariate = tf.data.Dataset.from_tensor_slices((val['x'],np.array(val['y'].reshape(-1,hyparams['pred_seq']*4), dtype=np.float32)))
    val_univariate = val_univariate.cache().shuffle(buffer_size).batch(batch_size)

    return train_univariate, val_univariate
