docker run -it --rm \
  --gpus all \
  --name  anoma\ # This is the container name
  --hostname $(hostname) \
  -e HOME \
  -p 8888:8888 \
  -u $(id -u):$(id -g) \
  -v $HOME:$HOME \
  -v /mnt/roahm:/mnt/roahm \
  -v $HOME/git/anomalous_pred:/tf/notebooks \
  anomalous_pred:jupyter
