# BiPOCO: Bi-directional Trajectory Prediction with Pose Constraints for Pedestrian Anomaly Detection

## Clone Repo
git clone --recurse-submodules https://github.com/akanuasiegbu/BiPOCO.git



## Docker Usage
1) cd into docker
2) run ./build.sh
3) use ./run.sh to enter docker file


## Requirements





## To ealutate AUC score
There is no training done with the part, we only evalute the given pkl file outputs. Training and inference is done with the [predictor model](https://github.com/akanuasiegbu/bitrap)

* In ```config/config.py``` change ```input_seq``` and ```pred_seq``` to match input and output sequence length.
* Also in ```config/config.py``` make sure to change ```exp['data']``` to match ```hr-st```, ```st```, ```avenue``` or ```hr-avenue```
* Also in ```config/config.py``` make sure to change ```exp['errortype']``` to match ```error_summed``` or ```error_flattened```
###### To run one file at a time
In ```experiments_code/main.py``` change variable ```file_to_load``` to point to correct pkl file.

##### To run multiple pkl files at time
Look at ```experiments_code/run_q.py```




