"""# Note that when I ran this orginally was using jupyter notebook
# WOuld need to make sure right directory is being looked looked at
# Also would want to add directoory and want correct directory to save new
# videos """

"""
loc : Directory that is looked at. (It's two directory out)
save_vid_loc : Directory that video is saved too
"""

from os import listdir
from os.path import isfile, join, isdir
import cv2
# loc = '/home/akanu/Dataset/Anomaly/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
loc = None
save_vid_loc = None # make
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

        out = cv2.VideoWriter( join(save_vid_loc, '{}.avi'.format(f)), cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

        for i in range(0,len(all_frames)):
            out.write(all_frames[i])
        out.release()
