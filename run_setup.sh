# setup environment
conda env create -f env.yaml
# install source
pip install -e .
# install ViTPose
conda run -n slahmr5 --live-stream pip install -v -e third_party/PHALP_plus/ViTPose
# install DROID-SLAM
cd third_party/DROID-SLAM
conda run -n slahmr5 --live-stream python setup.py install
cd ../..

# download models
gdown https://drive.google.com/uc?id=1GXAd-45GzGYNENKgQxFQ4PHrBp8wDRlW
unzip -q slahmr_dependencies.zip
rm slahmr_dependencies.zip
