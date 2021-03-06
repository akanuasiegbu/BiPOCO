"""
# Note that when I ran this orginally was using jupyter notebook
# WOuld need to make sure right directory is being looked looked at
# Also would want to add directoory and want correct directory to save new
# videos """

"""
loc : Directory that is looked at. (It's two directory out)
save_vid_loc : Directory that video is saved too
"""

import os
from os import listdir
from os.path import isfile, join, isdir
import cv2
from load_data import Files_Load
from config.config import hyparams, loc, exp

def vid_to_frames(vid_loc,  pic_loc):
    """
    Thats the location of the videos
    """

    frame_index = 0
    video_capture = cv2.VideoCapture(vid_loc)
    while True:
    
        ret, frame = video_capture.read()
        if ret != True:
            break

        cv2.imwrite(pic_loc + '/' +'{:04d}.jpg'.format(frame_index) , frame)
        frame_index += 1



def convert_spec_frames_to_vid(visual_plot_loc, save_vid_loc, vid_name, frame_rate):
    """
    visual_plot_loc : Directory that is looked at. 
    save_vid_loc : Directory that video is saved too
    frame_rate: frame rate for produced video
    """
    all_frames = []
    # loop over all the images in the folder
    for c in sorted(listdir(visual_plot_loc)):
        # if c[0] not in ['0', '1', '2', '3','4', '5','6','7','8','9']:
                # continue
        img_path = join(visual_plot_loc, c)
        img = cv2.imread(img_path)
        height,width,layers = img.shape
        size = (width, height)
        all_frames.append(img)

    out = cv2.VideoWriter( join(save_vid_loc, '{}.avi'.format(vid_name)), cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)

    for i in range(0,len(all_frames)):
        out.write(all_frames[i])
    out.release()

def conver_data_vid(loc, save_vid_loc):
    """
    Used this orginally in jupyter to convert ucsd dataset to videos
    loc : Directory that is looked at. (It's two directory out)
    save_vid_loc : Directory that video is saved too
    """

    # loc = '/home/akanu/Dataset/Anomaly/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
    # loc = None
    # save_vid_loc = None # make
    for f in sorted(listdir(loc)):
        # if f[0] != 'T' or f[-2:] == 'gt':
            # continue
        directory_path = join(loc, f)
        if isdir(directory_path):
            all_frames = []
            # loop over all the images in the folder
            for c in sorted(listdir(directory_path)):
                # if c[0] not in ['0', '1', '2', '3','4', '5','6','7','8','9']:
                        # continue
                img_path = join(directory_path, c)
                img = cv2.imread(img_path)
                height,width,layers = img.shape
                size = (width, height)
                all_frames.append(img)

            out = cv2.VideoWriter( join(save_vid_loc, '{}.avi'.format(f)), cv2.VideoWriter_fourcc(*'XVID'), 24, size)

            for i in range(0,len(all_frames)):
                out.write(all_frames[i])
            out.release()

def make_dir(dir_list, path):
    folder = join( path, *dir_list )

    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    
    return False



if __name__ =='__main__':
    path = '/mnt/roahm/users/akanu/dataset/Anomaly/ShangaiuTech/train_known_codec/frames_of_vid/train'
    save_vid = '/mnt/roahm/users/akanu/dataset/Anomaly/ShangaiuTech/train_known_codec/videos'
    conver_data_vid(loc=path, save_vid_loc=save_vid)
    # file = '/mnt/workspace/datasets/shanghaitech/training/videos'
    # test = False


    # for vid in sorted(listdir(file)):
    #     print(vid)

    #     if test:
    #         # dir_list = ['frames_of_vid', 'test', '{:02d}'.format(int(vid[:-4]))]
    #         dir_list = ['frames_of_vid', 'test', '{}'.format(vid[:-4])]
    #     else:
    #         # dir_list = ['frames_of_vid', 'train', '{:02d}'.format(int(vid[:-4]))]
    #         dir_list = ['frames_of_vid', 'train', '{}'.format(vid[:-4])]

    #     vid_loc = file + '/' + vid

    #     path = '/mnt/roahm/users/akanu/dataset/Anomaly/ShangaiuTech/train_known_codec/'
    #     make_dir(dir_list, path)
    #     pic_loc = join( path, *dir_list )

    #     vid_to_frames(vid_loc, pic_loc)    