#!/usr/bin/env bash
set -e

echo "Creating virtual environment"
python3.10 -m venv .slahmr_env
echo "Activating virtual environment"

source $PWD/.slahmr_env/bin/activate

# install pytorch
$PWD/.slahmr_env/bin/pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117

# torch-scatter
$PWD/.slahmr_env/bin/pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# install PHALP
$PWD/.slahmr_env/bin/pip install phalp[all]@git+https://github.com/brjathu/PHALP.git

# install source
$PWD/.slahmr_env/bin/pip install -e .

# install remaining requirements
$PWD/.slahmr_env/bin/pip install -r requirements.txt

# install ViTPose
$PWD/.slahmr_env/bin/pip install -v -e third-party/ViTPose

# install DROID-SLAM
cd third-party/DROID-SLAM
$PWD/../../.slahmr_env/bin/python setup.py install
cd ../..
