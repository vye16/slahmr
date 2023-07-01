import os
import subprocess

import numpy as np

import slahmr.preproc.launch_phalp as phalp
from slahmr.preproc.extract_frames import video_to_frames
from slahmr.preproc.launch_slam import (check_intrins, get_command,
                                        split_frames_shots)


def is_nonempty(d):
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def preprocess_frames(img_dir, src_path, overwrite=False, **kwargs):
    if not overwrite and is_nonempty(img_dir):
        print(f"FOUND {len(os.listdir(img_dir))} FRAMES in {img_dir}")
        return
    print(f"EXTRACTING FRAMES FROM {src_path} TO {img_dir}")
    print(kwargs)
    out = video_to_frames(src_path, img_dir, overwrite=overwrite, **kwargs)
    assert out == 0, "FAILED FRAME EXTRACTION"


def preprocess_tracks(img_dir, track_dir, shot_dir, overwrite=False):
    """
    :param img_dir
    :param track_dir, expected format: res_root/track_name/sequence
    :param shot_dir, expected format: res_root/shot_name/sequence
    """
    if not overwrite and is_nonempty(track_dir):
        print(f"FOUND TRACKS IN {track_dir}")
        return

    print(f"RUNNING PHALP ON {img_dir}")
    track_root, seq = os.path.split(track_dir.rstrip("/"))
    res_root, track_name = os.path.split(track_root)
    shot_name = shot_dir.rstrip("/").split("/")[-2]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", 0)

    phalp.process_seq(
        [gpu],
        seq,
        img_dir,
        f"{res_root}/phalp_out",
        track_name=track_name,
        shot_name=shot_name,
        overwrite=overwrite,
    )


def preprocess_cameras(cfg, overwrite=False):
    if not overwrite and is_nonempty(cfg.sources.cameras):
        print(f"FOUND CAMERAS IN {cfg.sources.cameras}")
        return

    print(f"RUNNING SLAM ON {cfg.seq}")
    img_dir = cfg.sources.images
    map_dir = cfg.sources.cameras
    subseqs, shot_idcs = split_frames_shots(cfg.sources.images, cfg.sources.shots)
    shot_idx = np.where(shot_idcs == cfg.shot_idx)[0][0]
    # run on selected shot
    start, end = subseqs[shot_idx]
    if not cfg.split_cameras:
        # only run on specified segment within shot
        end = start + cfg.end_idx
        start = start + cfg.start_idx
    intrins_path = cfg.sources.get("intrins", None)
    if intrins_path is not None:
        intrins_path = check_intrins(cfg.type, cfg.root, intrins_path, cfg.seq, cfg.split)

    cmd = get_command(
        img_dir,
        map_dir,
        start=start,
        end=end,
        intrins_path=intrins_path,
        overwrite=overwrite,
    )
    print(cmd)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
    out = subprocess.call(f"CUDA_VISIBLE_DEVICES={gpu} {cmd}", shell=True)
    assert out == 0, "SLAM FAILED"
