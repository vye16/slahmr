# Decoupling Human and Camera Motion from Videos in the Wild

Official PyTorch implementation of the paper Decoupling Human and Camera Motion from Videos in the Wild

[Project page](https://vye16.github.io/slahmr/)

[ArXiv](https://vye16.github.io/slahmr/)

<img src="./teaser.png">

## Getting started
This code was tested on Ubuntu 22.04 LTS and requires a CUDA-capable GPU

1. Clone repository and submodules
```
git clone --recursive https://github.com/vye16/slahmr.git
```
or initialize submodules if already cloned
```
git submodule update --init --recursive
```

2. Set up conda environment (note that creating the environment can take a while, especially the pip installation step has no feedback and can look like its stuck)
```
conda env create -f env.yaml
conda activate slahmr
```

Install current source repo
```
pip install -e .
```

Install ViTPose
```
pip install -v -e third_party/PHALP_plus/ViTPose
```

and DROID-SLAM (will take a while)
```
cd third_party/DROID-SLAM
python setup.py install
```

3. Download models from [here](https://drive.google.com/file/d/1GXAd-45GzGYNENKgQxFQ4PHrBp8wDRlW/view?usp=sharing).
```
gdown https://drive.google.com/uc?id=1GXAd-45GzGYNENKgQxFQ4PHrBp8wDRlW
unzip -q slahmr_dependencies.zip
rm slahmr_dependencies.zip
```

## Data
We provide configurations for dataset formats in `slahmr/confs/data`:
1. Posetrack in `slahmr/confs/data/posetrack.yaml`
2. Egobody in `slahmr/confs/data/egobody.yaml`
3. 3DPW in `slahmr/confs/data/3dpw.yaml`
4. DAVIS in `slahmr/confs/data/davis.yaml`
5. Custom video in `slahmr/confs/data/video.yaml`

**Please make sure to update all paths to data in the config files.**

We include tools to both process existing datasets we evaluated on in the paper, and to process custom data and videos.
We include experiments from the paper on the Egobody, Posetrack, and 3DPW datasets.

If you want to run on a large number of videos, or if you want to select specific people tracks for optimization,
we recommend preprocesing in advance. 
For a single downloaded video, there is no need to run preprocessing in advance.

From the `slahmr/preproc` directory, run PHALP on all your sequences
```
python launch_phalp.py --type <DATASET_TYPE> --root <DATASET_ROOT> --split <DATASET_SPLIT> --gpus <GPUS>
```
and run DROID-SLAM on all your sequences
```
python launch_slam.py --type <DATASET_TYPE> --root <DATASET_ROOT> --split <DATASET_SPLIT> --gpus <GPUS>
```
You can also update the paths to datasets in `slahmr/preproc/datasets.py` for repeated use.

## Run the code
Make sure all checkpoints have been unpacked `_DATA`.
We use hydra to launch experiments, and all parameters can be found in `slahmr/confs/config.yaml`.
If you would like to update any aspect of logging or optimization tuning, update the relevant config files.

From the `slahmr` directory,
```
python run_opt.py data=<DATA_CFG> run_opt=True run_vis=True
```

We've provided a helper script `launch.py` for launching many optimization jobs in parallel.
You can specify job-specific arguments with a job spec file, such as the example files in `job_specs`,
and batch-specific arguments shared across all jobs as
```
python launch.py --gpus 1 2 -f job_specs/pt_val_shots.txt -s data=posetrack exp_name=posetrack_val
```

We've also provided a separate `run_vis.py` script for running visualization in bulk.

In addition you can get an interactive visualization of the optimization procedure and the final output using [Rerun](https://github.com/rerun-io/rerun) with `python run_rerun_vis.py --log_root <LOG_DIR>`.

## BibTeX

If you use our code in your research, please cite the following paper:
```
@inproceedings{ye2023slahmr,
    title={Decoupling Human and Camera Motion from Videos in the Wild},
    author={Ye, Vickie and Pavlakos, Georgios and Malik, Jitendra and Kanazawa, Angjoo},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2023}
}
```
