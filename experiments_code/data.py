from load_data import Files_Load, Boxes, test_split_norm_abnorm, norm_train_max_min
from pedsort import pedsort
from load_data_binary_class import return_indices, binary_data_split, same_ratio_split_train_val

from sklearn.model_selection import train_test_split

from config import hyparams
import tensorflow as tf

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

def tensorify(X, Y):

    """
    Mainly using this function to make training and validation sets
    X: numpy array of training x for bounding boxes
    Y: numpy array of training y for bounding boxes

    return
    train_univariate: training tensor set
    val_univariate: validation tensor set
    """

    buffer_size = hyparams['buffer_size']
    batch_size = hyparams['batch_size']
    xx_train, xx_val,yy_train,yy_val = train_test_split(X,
                                                        Y,
                                                        test_size = hyparams['val_size'])
    train_univariate = tf.data.Dataset.from_tensor_slices((xx_train,yy_train))
    train_univariate = train_univariate.cache().shuffle(buffer_size).batch(batch_size)
    val_univariate = tf.data.Dataset.from_tensor_slices((xx_val,yy_val))
    val_univariate = val_univariate.cache().shuffle(buffer_size).batch(batch_size)

    return train_univariate, val_univariate
