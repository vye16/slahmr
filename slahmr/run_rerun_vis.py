"""Visualize SLAHMR results with rerun."""

import os
from typing import List, Optional

import numpy as np
import pytorch3d.structures
import rerun as rr
from rerun.components import Material
import torch
from matplotlib import colormaps
from omegaconf import OmegaConf
from scipy.spatial import transform

from slahmr.body_model import run_smpl
from slahmr.data import dataset, expand_source_paths, get_dataset_from_cfg
from slahmr.run_vis import get_input_dict, get_results_paths, load_result
from slahmr.util.loaders import (
    load_config_from_log,
    load_smpl_body_model,
    resolve_cfg_paths,
)
from slahmr.util.tensor import get_device, move_to

# define mapping from integer to RGB
_index_to_color = lambda x, cmap="tab10": colormaps[cmap](x % colormaps[cmap].N)


def log_to_rerun(
    cfg: dict,
    dataset: dataset.MultiPeopleDataset,
    log_dir: str,
    dev_id: int,
    phases: List[str] = ["motion_chunks"],
    phase_labels: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    assert phase_labels is None or len(phases) == len(phase_labels)

    if len(dataset) < 1:
        print("No tracks in dataset, skipping")
        return

    if phase_labels is None:
        phase_labels = [f"{i}_{p}" for i, p in enumerate(phases)]

    rr.init("slahmr", spawn=save_dir is None)
    if save_dir is not None:
        rr.save(os.path.join(save_dir, "log.rrd"))

    # NOTE: first camera view defines world coordinate system
    #  assuming camera is upright, and following RDF convention, Y will be down
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    dataset.load_data()

    for phase, phase_label in zip(phases, phase_labels):
        log_pinhole_camera(dataset, phase_label)
        log_input_frames(dataset, phase_label)
        log_skeleton_2d(dataset, phase_label)

        phase_dir = os.path.join(log_dir, phase)
        if phase == "input":
            res = get_input_dict(dataset)
            it = f"{0:06d}"

        elif os.path.isdir(phase_dir):
            res_path_dict = get_results_paths(phase_dir)
            it = sorted(res_path_dict.keys())[-1]
            res = load_result(res_path_dict[it])["world"]
        else:
            print(f"{phase_dir} does not exist, skipping")
            continue

        log_phase_result(cfg, dataset, dev_id, phase, phase_label, res)


def log_pinhole_camera(dataset: dataset.MultiPeopleDataset, phase_label: str) -> None:
    """Log camera trajectory to rerun."""
    cam_data = dataset.get_camera_data()
    fx, fy, cx, cy = cam_data["intrins"][0]
    width, height = dataset.img_size
    rr.set_time_sequence(f"frame_id_{phase_label}", 0)
    rr.set_time_sequence("frame_id", 0)
    rr.log(
        f"world/{phase_label}/camera/image",
        rr.Pinhole(
            image_from_camera=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            width=width,
            height=height,
        ),
    )
    rr.set_time_sequence(f"frame_id_{phase_label}", None)


def log_phase_result(
    cfg,
    dataset: dataset.MultiPeopleDataset,
    dev_id,
    phase: str,
    phase_label: str,
    phase_result: dict,
) -> None:
    """Log results from one phase."""
    B = len(dataset)
    num_frames = dataset.seq_len
    vis_mask = dataset.data_dict["vis_mask"]  # -1 out of frame, 0 occluded, 1 visible
    device = get_device(dev_id)
    phase_result = move_to(phase_result, device)

    cfg = resolve_cfg_paths(cfg)
    body_model, _ = load_smpl_body_model(cfg.paths.smpl, B * num_frames, device=device)

    with torch.no_grad():
        world_smpl = run_smpl(
            body_model,
            phase_result["trans"],
            phase_result["root_orient"],
            phase_result["pose_body"],
            phase_result.get("betas", None),
        )

    # compute vertex normals on GPU
    num_tracks = world_smpl["vertices"].shape[0]
    num_meshes = num_tracks * num_frames
    num_vertices = world_smpl["vertices"].shape[-2]
    num_faces = world_smpl["faces"].shape[-2]

    # faces has no batch dim here, use expand to avoid copy
    meshes = pytorch3d.structures.Meshes(
        verts=world_smpl["vertices"].reshape(num_meshes, num_vertices, 3),
        faces=world_smpl["faces"].expand(num_meshes, num_faces, 3),
    )

    vertices = world_smpl["vertices"].numpy(force=True)
    faces = world_smpl["faces"].numpy(force=True)
    vertex_normals = (
        meshes.verts_normals_padded()
        .reshape(num_tracks, num_frames, num_vertices, 3)
        .numpy(force=True)
    )

    # NOTE: if the meshes don't deform over time or are similar we could use the same
    #  vertex normals for all frames and/or tracks

    for frame_id in range(num_frames):
        rr.set_time_sequence(f"frame_id_{phase_label}", frame_id)
        rr.set_time_sequence("frame_id", frame_id)
        translation = phase_result["cam_t"][1, frame_id].numpy(force=True)
        rotation_mat = phase_result["cam_R"][1, frame_id].numpy(force=True)
        rotation_q = transform.Rotation.from_matrix(rotation_mat).as_quat()
        rr.log(
            f"world/{phase_label}/camera",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation_q),
                from_parent=True,
            ),
        )
        for i, _ in enumerate(dataset.track_ids):
            if vis_mask[i][frame_id] >= 0:
                rr.log(
                    f"world/{phase_label}/#{i}",
                    rr.Mesh3D(
                        vertex_positions=vertices[i, frame_id],
                        indices=faces,
                        vertex_normals=vertex_normals[i, frame_id],
                        mesh_material=Material(albedo_factor=_index_to_color(i)),
                    )
                )
            else:
                rr.log(f"world/{phase_label}/#{i}", rr.Clear(recursive=True))
    rr.set_time_sequence(f"frame_id_{phase_label}", None)


def log_input_frames(dataset: dataset.MultiPeopleDataset, phase_label: str) -> None:
    """Log raw input video to rerun."""
    for frame_id, img_path in enumerate(dataset.sel_img_paths):
        rr.set_time_sequence(f"frame_id_{phase_label}", frame_id)
        rr.set_time_sequence("frame_id", frame_id)
        rr.log(f"world/{phase_label}/camera/image", rr.ImageEncoded(path=img_path))
    rr.set_time_sequence(f"frame_id_{phase_label}", None)


def log_skeleton_2d(dataset: dataset.MultiPeopleDataset, phase_label: str) -> None:
    """Log 2D skeleton to rerun."""
    # see vis.tools.vis_keypoints and ViTPose/mmpose/apis/inference.py
    SKELETON_IDS = np.array(
        [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ]
    )
    # see vis.tools.imshow_keypoints
    IDCS = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
    for i, _ in enumerate(dataset.track_ids):
        joints2d = dataset.data_dict["joints2d"][i]  # (T, J, 3)
        for frame_id, frame_joints in enumerate(joints2d):
            joints = frame_joints[IDCS][SKELETON_IDS]
            joint_confidence = joints[..., 2].min(axis=-1)  # min conf per joint
            good_joints_xy = joints[joint_confidence > 0.3, :, :2]

            rr.set_time_sequence(f"frame_id_{phase_label}", frame_id)
            rr.set_time_sequence("frame_id", frame_id)
            if len(good_joints_xy):
                rr.log(
                    f"world/{phase_label}/camera/image/skeleton/#{i}",
                    rr.LineStrips2D(
                        good_joints_xy,
                        colors=_index_to_color(i),
                    ),
                )
            else:
                rr.log(
                    f"world/{phase_label}/camera/image/skeleton/#{i}",
                    rr.Clear(recursive=True)
                )
    rr.set_time_sequence(f"frame_id_{phase_label}", None)


def main(args):
    """
    visualize all runs in root
    """
    OmegaConf.register_new_resolver("eval", eval)
    log_dirs = []
    for root, subd, files in os.walk(args.log_root):
        if ".hydra" in subd:
            log_dirs.append(root)
    args.log_dirs = log_dirs
    print(f"FOUND {len(args.log_dirs)} TO RENDER")

    for log_dir in args.log_dirs:
        dev_id = 0
        path_name = log_dir.split(args.log_root)[-1].strip("/")
        exp_name = "-".join(path_name.split("/")[:2])
        cfg = load_config_from_log(log_dir)
        cfg.data.sources = expand_source_paths(cfg.data.sources)
        print("SOURCES", cfg.data.sources)
        dataset = get_dataset_from_cfg(cfg)

        save_dir = None
        if args.save_root:
            save_dir = f"{args.save_root}/{exp_name}"
            os.makedirs(save_dir, exist_ok=True)

        log_to_rerun(
            cfg,
            dataset,
            log_dir,
            dev_id,
            phases=args.phases,
            phase_labels=args.phase_labels,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--save_root", default=None)
    parser.add_argument(
        "--phases",
        nargs="*",
        default=["init", "root_fit", "smooth_fit", "motion_chunks"],
    )
    parser.add_argument("--phase_labels", nargs="*", default=None)
    parser.add_argument("--gpus", nargs="*", default=[0])
    args = parser.parse_args()

    main(args)
