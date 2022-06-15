# BiPOCO: Bi-directional Trajectory Prediction with Pose Constraints for Pedestrian Anomaly Detection

## Clone Repo
git clone --recurse-submodules https://github.com/akanuasiegbu/BiPOCO.git



## Docker Usage
1) cd into docker
2) run ./build.sh
3) use ./run.sh to enter docker file


## Requirements

## Step 1: Pose Data Input into BiTRAP
* The inputted data into BiTrap for train and test poses can be found in this [folder](https://drive.google.com/drive/folders/1oNKUXdYlNP1g7M9T3E1UWERh0lFobKAl?usp=sharing).
  * Next download the json files and put them in a folder. Then in ```bitrap/datasets/config_for_my_data.py``` set ```loc['data_load']['avenue']['train_poses']```.   ```loc['data_load']['avenue']['test_poses']```,  ```loc['data_load']['st']['train_poses']```, and  ```loc['data_load']['st']['test_poses']``` to the correct directory.
* To recreate the pose input data
  * Run [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose/tree/ddaf4b99327132f7617a768a75f7cb94870ed57c) (commit number ddaf4b9)
    * Config file used was ```configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml```
    * Pretrained model used was ```pretrained_models/fast_res50_256x192.pth```
    * Tracker used was Human-ReID based tracking (```--pose_track```)
  * Next with json files from AlphaPose add the anomaly labels with the ```add_to_json_file.py``` for only the testing data


## Step 2: Training
##### Pose trajectory training on Avenue and ShanghaiTech Dataset

Users can train the BiTraP models on Avenue and ShanghaiTech dataset easily by runing the following command:

Train on Avenue Dataset
```
python bitrap/tools/train.py --config_file configs/avenue_pose_hc.yml
```

Train on ShanghaiTech Dataset
```
python  bitrap/tools/train.py --config_file configs/st_pose_hc.yml
```

To train/inferece on CPU or GPU, simply add `DEVICE='cpu'` or  `DEVICE='cuda'`. By default we use GPU for both training and inferencing.

Note that you must set the input and output lengths to be the same in YML file used (```INPUT_LEN``` and ```PRED_LEN```) and ```bitrap/datasets/config_for_my_data.py``` (```input_seq``` and ```pred_seq```)

## Step 3: Inference 

##### Pretrained Models
Pretrained models for [Avenue](https://drive.google.com/drive/folders/1ra1XTB8KpBOy7Xgxg8of3DwjoIJyd9bV?usp=sharing) and [ShanghaiTech](https://drive.google.com/drive/folders/1-vY3MWPaWbwwgWOiOcD-sXXzqHidXYJv?usp=sharing) can found.

##### PKL Files
Pkl files of the best performing configuration bolded in table [2](https://drive.google.com/drive/folders/1jO3RnkvOsR-VLdATyzeMDsGF7mAu5Qdl?usp=sharing) and [3](https://drive.google.com/drive/folders/1ztgVn6Oq2Poq1PpAMzgL9yj00UToXn8K?usp=sharing) can be found.


##### Pose trajectory prediction on Avenue and ShanghaiTech Dataset
TO obtain the rest of the pkl files for the pose trajectory for first-person (ego-centric) view Avenue and ShanghaiTech datasets use commands below. 

Test on Avenue dataset:
```
python tools/test.py --config_file bitrap/configs/avenue_pose_hc.yml CKPT_DIR **DIR_TO_CKPT**

```

Test on ShanghaiTech dataset:
```
python tools/test.py --config_file bitrap/configs/st_pose_hc.yml CKPT_DIR **DIR_TO_CKPT**
```

Note that you must set the input and output lengths to be the same in YML file used (```INPUT_LEN``` and ```PRED_LEN```) and ```bitrap/datasets/config_for_my_data.py``` (```input_seq``` and ```pred_seq```)



## Step 4: To evaluate AUC score

Training and inference is done with the [predictor model](https://github.com/akanuasiegbu/bitrap). Given the PKL output files from inference we can obtain AUC score by following the instructions below. 

* In ```config/config.py``` change ```input_seq``` and ```pred_seq``` to match input and output sequence length.
* Also in ```config/config.py``` make sure to change ```exp['data']``` to match ```hr-st```, ```st```, ```avenue``` or ```hr-avenue```
* Also in ```config/config.py``` make sure to change ```exp['errortype']``` to match ```error_summed``` or ```error_flattened```
###### To run one file at a time
In ```experiments_code/main.py``` change variable ```file_to_load``` to point to correct pkl file.

##### To run multiple pkl files at time
Look at ```experiments_code/run_q.py```




