import json 
from scipy.io import loadmat
import numpy as np
from skimage.measure import label, regionprops
from custom_functions.visualizations_pose import vis_frame, Opt
import cv2
"""
Aim to append to json file 
1) Append frame level abnormal label. check
2) Add human level abnormal label. check
3) Visulaiztion for human level abnormal label for verification 
"""


# Import json files 

def json_add_frame_level(traj_loc, mask_loc, vid_num, dataset ='avenue'):
    res = json.load(traj_loc)
    mask_data = loadmat(mask_loc)
    frame_level = []
    
    for mask in mask_data['volLabel'][0]:
        if np.all(mask == 0):
            frame_level.append(0)
        else:
            frame_level.append(1)

    # image id starts at 0 and the frame_level array index starts at 0 
    frame_level = np.array(frame_level) 
    
    
    # Cycle through and long as frame is the same 
    image_id_prev = 0
    abnormal_ped = []
    temp_frame_bbox = []
    for pedestrain in res:
        image_id_next = pedestrain['image_id']
        if image_id_next == image_id_prev:
            temp_frame_bbox.append(pedestrain['box'])
        else:
            if dataset == 'avenue':
                mask = mask_data['volLabel'][0,image_id_prev]
            
            abnormal_ped.append(find_anomaly(mask, temp_frame_bbox))
            temp_frame_bbox = []
            temp_frame_bbox.append(pedestrain['box'])
            
        image_id_prev = image_id_next
    
    # accounting for last
    if dataset == 'avenue':
        mask = mask_data['volLabel'][0,image_id_prev]
        abnormal_ped.append(find_anomaly(mask, temp_frame_bbox))
    
    abnormal_ped = np.concatenate(abnormal_ped)
    
    # Need to go into list and add dict format
    # cycle through to detect abnormal frames
    temp_list = []
    for count, pedestrain in enumerate(res):
        image_id = pedestrain['image_id']
        pedestrain['abnormal_gt_frame'] = int(frame_level[image_id])
        pedestrain['abnormal_pedestrain'] = int(abnormal_ped[count])
        temp_list.append(pedestrain)
    
    print('here')
    
    with open('../output_pose_json_appended/{:02d}_append.json'.format(vid_num), 'w') as json_file:
        json.dump(temp_list, json_file )

def find_anomaly(mask_data, temp_frame_bbox):
    """
    mask_data:  
    bbox: 
    would be nice to pass bbox of entire frame
    """
    iou_limit = 0.5
    abnormal_ped = np.zeros(len(temp_frame_bbox))
    
    if np.all(mask_data == 0):
        # Need to add return here
        return abnormal_ped
    
    abnorm_bbox = regionprops(label(mask_data)) # This returns a list
    
    abnorm_iou_matrix = []
    for bbox in temp_frame_bbox:
        # comes in as tlwh _> tlbr
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]#xmin,xmax,ymin,ymax
        found_iou = []
        for abnorm in abnorm_bbox:
            mask_box = [abnorm.bbox[1],abnorm.bbox[0], abnorm.bbox[3],abnorm.bbox[2] ]
            iou = bb_intersection_over_union(mask_box, bbox)
            found_iou.append(iou)
        # Basically save iou for diff bbox as row
        abnorm_iou_matrix.append(np.array(found_iou))
    
    # rows will correspond to index to change    
    abnorm_iou_matrix = np.array(abnorm_iou_matrix)

    # Finds the max iou for each person
    abnorm_iou_array = np.amax(abnorm_iou_matrix, axis=1)
    # abnorm_iou_array = np.array(abnorm_iou_array)

    if len(abnorm_bbox) == 1:
        abnorm_iou_array = abnorm_iou_array.reshape((-1,1))
        row, col =np.where(abnorm_iou_array == np.amax(abnorm_iou_array))
        max_index = np.array(row)
        # print('here where len is 1')
    else:    
        # want iou to be greater than iou_limit and constraint that largest amount of people
        # labeled abnormal is equal to max amount of abnormal bbox found 

        
        # Finds iou greater than iou_limit
        row = np.where(abnorm_iou_array >= iou_limit)
        row = np.array(row).reshape(-1,1)
        # print('row shape before {}, {}'.format(row, row.shape))
        if row.shape == (0,1):
            # print('Not at least iou:{}'.format(iou_limit))
            return abnormal_ped

        # creates a temp array and appends index value to column
        temp_iou = abnorm_iou_array[row].reshape(-1,1)
        # print('temp iou shape before {}, {}'.format(temp_iou,temp_iou.shape))

        temp_iou = np.append(temp_iou, row, axis =1)
        
        # print('temp iou after {},{}'.format(temp_iou, temp_iou.shape))
        # If percieved amount of abnormality is greater than i
        assert temp_iou.shape == (row.shape[0], 2)
        
        if len(temp_iou[:,1]) > len(temp_frame_bbox):
            # arg sort based on the iou
            index_sorted_iou = np.argsort(temp_iou[:,0]) #sorts from min to max
            row = temp_iou[:,1][index_sorted_iou][-len(temp_frame_bbox):]
            max_index = np.array(row)
            # print('Here I am')
        else:
            max_index = np.array(row)
            # print("len is less than temp")

    # Save Index and max iou
    # Set index out of bound every time reset -1
    max_iou = np.max(abnorm_iou_array)

    if max_iou != 0.0:
        # avoids error when max_iou = 0.0
        for per in max_index:
            # print(int(per))
            abnormal_ped[int(per)] = 1
            # temp_frame_bbox[int(per)].anomaly = 1
    
    return abnormal_ped
    
    
def plot_appended_json_file(traj_loc, vid_num):
    
    res = json.load(traj_loc)
    image_id_prev = 0
    temp_ped = []
    imgs = []
    
    for pedestrain in res:
        image_id_next = pedestrain['image_id']
        
        if image_id_next == image_id_prev:
            pedestrain['keypoints'] = np.array(pedestrain['keypoints']).reshape(17,3)
            temp_ped.append(pedestrain)
        else:
            # call plotting function
            result_partial = { 'imgname':image_id_prev , 'result': temp_ped } 
            frame = '/mnt/roahm/users/akanu/dataset/Anomaly/Avenue_Dataset/frames_of_vid/test/{:02d}/{:02d}.jpg'.format(vid_num, image_id_prev)
            frame = cv2.imread(frame)

            imgs.append(vis_frame(frame, result_partial, Opt))
            
            temp_ped = []
            pedestrain['keypoints'] = np.array(pedestrain['keypoints']).reshape(17,3)
            temp_ped.append(pedestrain)
            
        image_id_prev = image_id_next
    
    imgs.append(vis_frame(frame, result_partial, Opt))
    

    height, width, layers = frame.shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_vid = '../output_pose_json_appended/{:02d}.avi'.format(vid_num)

    out = cv2.VideoWriter(save_vid, fourcc, 25, size)
    
    for i in imgs:
        out.write(i)

    out.release()

    


# Begin Added
def bb_intersection_over_union(boxA, boxB):
    # Found Online
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA ) * max(0, yB - yA )
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

    
if __name__ == "__main__":
    
    for i in range(1,22):
    # can for loop this and append the json file
        traj_loc = open('/mnt/roahm/users/akanu/projects/AlphaPose/avenue_alphapose_out/test/{}/alphapose-results.json'.format(i))
        mask_loc = '/mnt/workspace/datasets/avenue/ground_truth_demo/testing_label_mask/{:1d}_label.mat'.format(i)
        json_add_frame_level(traj_loc, mask_loc, vid_num = i, dataset = 'avenue')
    
    # # plot video to check
    # for i in range(1,22):
    #     json_append_loc= open('../output_pose_json_appended/{:02d}_append.json'.format(i))
    #     plot_appended_json_file(json_append_loc, i)