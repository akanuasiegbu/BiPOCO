# BiPOCO: Bi-directional Trajectory Prediction with Pose Constraints for Pedestrian Anomaly Detection

## Clone Repo
git clone --recurse-submodules https://github.com/akanuasiegbu/BiPOCO.git



## Docker Usage
1) cd into docker
2) run ./build.sh
3) use ./run.sh to enter docker file


## Requirements













Credits

Naming convention of saved models

model: lstm
type: xywh, tlbr
dataset: ped1,ped2,st, avenue
seq: size of sequence, 20, 5 etc int of sequence

model_type_dataset_seq.h5
example: lstm_xywh_ped1_20.h5

1) Generate trajactery data 
2) Run BiTrap and LSTM
3) Run the Main file
4) Change the parameters of config.py 
