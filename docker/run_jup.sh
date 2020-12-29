docker run -it --rm \
  --gpus all \
  --name  anoma\
  --hostname $(hostname) \
  -e HOME \
  -p 8888:8888 \
  -u $(id -u):$(id -g) \
  -v $HOME:$HOME \
  -v /mnt/roahm:/mnt/roahm \
  -v /mnt/workspace/datasets:/mnt/workspace/datasets:ro \
  -v $HOME/akanu/projects/anomalous_pred:/tf/notebooks \
  anomalous_pred:jupyter
