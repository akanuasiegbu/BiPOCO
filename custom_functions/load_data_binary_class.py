import numpy as np
from math import floor
from custom_metrics import bb_intersection_over_union, bb_intersection_over_union_np
from coordinate_change import xywh_tlbr, tlbr_xywh



def return_indices(data=None, abnormal_split = 0.5, seed=None):
    """
    Note that function returns index values that will
    allow for the creation of a train and test set that has an specificed ratio
    of normal and abnormal examples. Rest of abnormal and normal are then used
    in the training set.

    data: 1 and 0's whose location in index corresponds to location
            in acutal dataset
    abnormal_split: percentage of abnormal frames to put in test frame

    returns: list that contains indices
            [train_abn_indices, train_n_indices, test_abn_indices, test_n_indices]
    """

    np.random.seed(seed)

    abnorm_index = np.where(data == 1)
    norm_index = np.where(data == 0)

    rand_an = np.random.permutation(len(abnorm_index[0]))
    rand_n = np.random.permutation(len(norm_index[0]))
    # Permutates the found abnormal and normal indices
    abnorm_index = abnorm_index[0][rand_an]
    norm_index = norm_index[0][rand_n]


    len_abn_split = floor(len(abnorm_index)*abnormal_split)

    # Testing set indices
    test_abn_indices = abnorm_index[:len_abn_split]
    test_n_indices = norm_index[:len_abn_split]

    train_abn_indices = abnorm_index[len_abn_split:]
    train_n_indices = norm_index[len_abn_split:]

    return [train_abn_indices, train_n_indices, test_abn_indices, test_n_indices]



def binary_data_split(x,y, model, indices):
    """
    This function takes normed data and returns training data with IOU and return_indices
    Indices can be used to track back to location in unshuffled testdict.
    So that visulzations of what can be given as to what happened.

    x: normed testing data
    y: normed tested data
    indices: [train_abn_indices, train_n_indices, test_abn_indices, test_n_indices]

    return: train_x, train_y, test_x, test_y
            Note that the second coloumn of train_x and test_x
            contain the indices corresponding the location in unshuffled
            dictornary
    """
    # I dont like that I put model predict in here
    # function name is kind of deceiving
    out1 = model.predict(x)
    out = bb_intersection_over_union_np(xywh_tlbr(out1),xywh_tlbr(y))
    out = np.squeeze(out)

    train_x = np.array( [np.append(out[indices[0]], out[indices[1]] ),
                        np.append(indices[0], indices[1])])

    train_y = np.append(np.ones(len(indices[0])),
                  np.zeros(len(indices[1])) )


    test_x = np.array( [np.append(out[indices[2]], out[indices[3]] ),
                        np.append(indices[2], indices[3])])
    test_y = np.append(np.ones(len(indices[2])),
                      np.zeros(len(indices[3])) )

    return train_x,train_y, test_x, test_y





def same_ratio_split_train_val(train_x,train_y, val_ratio = 0.3):
    """
    train_x: training x data for binary classifer. shape (2, somenumber)
    train_y: training y data for binary classifer. shape (somenumber,)
    val_ratio: ratio to split between validation and training set

    return: val_x, val_y, train_x, train_y
    """

    abnorm_index = np.where(train_y == 1)
    norm_index = np.where(train_y == 0)
    rand_an = np.random.permutation(len(abnorm_index[0]))
    rand_n = np.random.permutation(len(norm_index[0]))
    abnorm_index = abnorm_index[0][rand_an]
    norm_index = norm_index[0][rand_n]

    len_val_ab = int(len(abnorm_index)*val_ratio)
    len_val_n = int(len(norm_index)*val_ratio)


    val_x = np.append(train_x[:,abnorm_index[:len_val_ab]] ,
                      train_x[:,norm_index[:len_val_n]], axis = 1)

#     part1 = train_x[:,abnorm_index[:len_val_ab]]
#     part2 = train_x[:,norm_index[:len_val_n]]
#     print('This is shape of part 1 of append: {}'.format(part1.shape))
#     print('This is shape of part 2 of append: {}'.format(part2.shape))
#     print('This is the combined shape of part 1 and 2 {}'.format(val_x.shape))

    val_y = np.append(train_y[abnorm_index[:len_val_ab]],
                      train_y[norm_index[:len_val_n]])

    train_x = np.append(train_x[:,abnorm_index[len_val_ab:]] ,
                      train_x[:,norm_index[len_val_n:]], axis=1)

    train_y = np.append(train_y[abnorm_index[len_val_ab:]],
                      train_y[norm_index[len_val_n:]])


    return val_x, val_y, train_x, train_y



def one_weight_ratio_train(train_x, train_y):
    """
    This function splits the training data to an equal amount of abnormal
    and normal sequences. Returns same type of data as inputted.
    rows and col are same format. Think about as removing the excess normal Values
    that are not used.

    train_x: this contains 2 rows. First row is the iou values.
             second row is the index values that correspond to locations in
             train dict.
    train_y: has indicator 1 or 0 depending on abnormal or not

    return: train_x_even_split, train_y_even_split
    """
    abnorm_index = np.where(train_y ==1)[0]
    norm_index = np.where(train_y == 0)[0]
    rand_norm = np.random.permutation(len(norm_index))
    norm_index = norm_index[rand_norm]
    train_x_even_split = np.append(train_x[:,abnorm_index],
                                train_x[:,norm_index][:,:len(abnorm_index)],
                                  axis=1)
    train_y_even_split = np.append(train_y[abnorm_index],
                                  train_y[norm_index][:len(abnorm_index)])
    return train_x_even_split, train_y_even_split
