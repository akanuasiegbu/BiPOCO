if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

docker run -it \
  -u $(id -u):$(id -g) \
  --rm \
  --gpus '"device='$GPU'"' \
  --hostname $(hostname) \
  -e HOME \
  -v $(pwd)/.bash_history:$HOME/.bash_history \
  -v /mnt/roahm:/mnt/roahm \
  -v /mnt/workspace/datasets:/mnt/workspace/datasets:ro \
  -w /home/akanu \
  -v $HOME/akanu/projects/Projects_to_make_public/BiPOCO:/home/akanu \
  bipoco:latest

