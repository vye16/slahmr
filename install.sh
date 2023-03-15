#!/usr/bin/bash
NAME=slahmr
conda create -n $NAME python=3.9 -y
conda run -n $NAME --live-stream conda install suitesparse -c conda-forge -y

# install pytorch using pip, update with appropriate cuda drivers if necessary
conda run -n $NAME --live-stream pip install torch==1.13.0 torchvision==1.14.0 --index-url https://download.pytorch.org/whl/cu117
# uncomment if pip installation isn't working
# conda run -n $NAME --live-stream conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
# install pytorch scatter using pip, update with appropriate cuda drivers if necessary
conda run -n $NAME --live-stream pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
# uncomment if pip installation isn't working
# conda run -n $NAME --live-stream conda install pytorch-scatter -c pyg -y

# install remaining requirements
conda run -n $NAME --live-stream pip install -r requirements.txt

# install source
conda run -n $NAME --live-stream pip install -e .

# install ViTPose
conda run -n $NAME --live-stream pip install -v -e third-party/PHALP_plus/ViTPose

# install DROID-SLAM
cd third-party/DROID-SLAM
conda run -n $NAME --live-stream python setup.py install
