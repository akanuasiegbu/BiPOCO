if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

# docker run -it --rm \
#   --gpus '"device='$GPU'"' \
#   --name anomalous_ser \ # This is the container name
#   --hostname $(hostname) \
#   -e HOME \
#   -p 8888:8888 \
#   -u $(id -u):$(id -g) \
#   -v $HOME:$HOME \
#   -v /mnt/roahm:/mnt/roahm \
#   #  -v $HOME/git/anomalous_pred:/tf/notebooks \
#   #-v $HOME/project/anomalous_pred:/tf/notebooks \
#   anomalous_pred_ser:latest

docker run -it \
  -u $(id -u):$(id -g) \
  --rm \
  --gpus '"device='$GPU'"' \
  --hostname $(hostname) \
  -e HOME \
  -p 80:8888 \
  -v $(pwd)/.bash_history:$HOME/.bash_history \
  -v /mnt/roahm:/mnt/roahm \
  -v /home/akanu/project/anomalous_pred:/home/akanu \
  -it \
   anomalous_pred_ser:latest

