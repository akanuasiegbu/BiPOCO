# CUDA 10.1 requires driver version >= 418.39
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

RUN apt-get update && \
    apt-get -y install apt-utils libopencv-dev cmake software-properties-common nano wget tree vim 





# RUN apt-get update && \
#     apt-get -y install apt-utils libopencv-dev cmake git sudo vim software-properties-common nano wget tree

RUN pip install matplotlib==3.4.1 tqdm==4.36.1 yacs==0.1.8 Pillow==7.0.0 tensorboardx==2.2 wandb==0.10.25 scikit-learn==0.24.1 opencv-python==4.5.1.48  coloredlogs==15.0 termcolor==1.1.0 dill==0.3.3
RUN pip install six==1.13.0

RUN pip install scipy==1.6.2 seaborn==0.11.0 pandas==1.1.2 more_itertools==8.8.0
RUN pip install protobuf==3.15.8 

# RUN pip install pandas numpy

# Avoid showing "I have no name!" in a container
RUN rm -f /etc/bash.bashrc && \
    wget -O /etc/bash.bashrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/dockerfiles/bashrc && \
    sed -i "s/tf-docker/bipoco/g" /etc/bash.bashrc && \
    sed -i "25,35d" /etc/bash.bashrc

# RUN export PYTHONPATH=$PWD:$PWD/custom_functions:PYTHONPATH

ENV PYTHONPATH "/mnt/roahm/users/akanu/projects/Projects_to_make_public/BiPOCO/custom_functions:/home/akanu"