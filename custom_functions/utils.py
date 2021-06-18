import os, sys, time
from os.path import join


def make_dir(dir_list):
    try:
        print(os.makedirs(join( os.path.dirname(os.getcwd()),
                                *dir_list )) )
    except OSError:
        print('Creation of the directory {} failed'.format( join(os.path.dirname(os.getcwd()),
                                                            *dir_list) ) )
    else:
        print('Successfully created the directory {}'.format(   join(os.path.dirname(os.getcwd()),
                                                                *dir_list) ) )


class SaveTextFile(object):
    def __init__(self, save_path):
        self._directory =  save_path

        # with open(self._directory, 'a') as with_as_write:
        #     # with_as_write.write('Frame_Number Person_ID State BB_tl_0 BB_tl_1 BB_br_0 BB_br_1 abnomaly\n')
        #     with_as_write.write('Frame_Number Person_ID State BB_tl_0 BB_tl_1 BB_br_0 BB_br_1 abnormal_ped abnormal_gt\n')

    def save(self, text, number):
        with open(self._directory, 'a') as with_as_write:
            with_as_write.write('{} {} \n'.format(text, number))
