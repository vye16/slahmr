import os

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from slahmr.data import get_dataset_from_cfg, expand_source_paths
from slahmr.optim.output import (
    get_results_paths,
    load_result,
    save_input_frames,
)
from slahmr.util.loaders import load_config_from_log, resolve_cfg_paths, load_smpl_body_model
from slahmr.util.tensor import get_device, move_to, to_torch
from slahmr.vis.output import (
    prep_result_vis,
    animate_scene,
    make_video_grid_2x2,
)
from slahmr.vis.tools import vis_keypoints
from slahmr.vis.viewer import init_viewer


def run_vis(
    cfg,
    dataset,
    out_dir,
    dev_id,
    phases=["motion_chunks"],
    render_views=["src_cam", "above", "side"],
    make_grid=False,
    overwrite=False,
    save_dir=None,
    render_kps=True,
    render_layers=False,
    save_frames=False,
    **kwargs,
):
    save_dir = out_dir if save_dir is None else save_dir
    print("OUT_DIR", out_dir)
    print("SAVE_DIR", save_dir)
    print("VISUALIZING PHASES", phases)
    print("RENDERING VIEWS", render_views)
    print("OVERWRITE", overwrite)

    # save input frames
    inp_vid_path = save_input_frames(
        dataset,
        f"{save_dir}/{dataset.seq_name}_input.mp4",
        fps=cfg.fps,
        overwrite=True,
    )

    if render_kps:
        render_keypoints_2d(dataset, save_dir, overwrite=overwrite)

    if len(render_views) < 1:
        return

    out_ext = "/" if render_layers or save_frames else ".mp4"
    phase_results = {}
    phase_max_iters = {}
    for phase in phases:
        res_dir = os.path.join(out_dir, phase)
        if phase == "input":
            res = get_input_dict(dataset)
            it = f"{0:06d}"

        elif os.path.isdir(res_dir):
            res_path_dict = get_results_paths(res_dir)
            it = sorted(res_path_dict.keys())[-1]
            res = load_result(res_path_dict[it])["world"]

        else:
            print(f"{res_dir} does not exist, skipping")
            continue

        out_name = f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}"
        phase_max_iters[phase] = it

        out_paths = [f"{out_name}_{view}{out_ext}" for view in render_views]
        if not overwrite and all(os.path.exists(p) for p in out_paths):
            print("FOUND OUT PATHS", out_paths)
            continue

        phase_results[phase] = out_name, res

    if len(phase_results) > 0:
        out_names, res_dicts = zip(*phase_results.values())
        render_results(
            cfg,
            dataset,
            dev_id,
            res_dicts,
            out_names,
            render_views=render_views,
            render_layers=render_layers,
            save_frames=save_frames,
            **kwargs,
        )

    if make_grid:
        for phase, it in phase_max_iters.items():
            grid_path = f"{save_dir}/{dataset.seq_name}_{phase}_grid.mp4"
            make_video_grid_2x2(
                grid_path,
                [
                    inp_vid_path,
                    f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}_src_cam.mp4",
                    f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}_above.mp4",
                    f"{save_dir}/{dataset.seq_name}_{phase}_final_{it}_side.mp4",
                ],
                overwrite=True,
            )


def get_input_dict(dataset):
    dataset.load_data(interp_input=False)
    d = dataset.data_dict
    input_params = {
        "pose_body": np.stack(d["init_body_pose"], axis=0),
        "trans": np.stack(d["init_trans"], axis=0),
        "root_orient": np.stack(d["init_root_orient"], axis=0),
    }
    input_params = to_torch(input_params)
    print({k: v.shape for k, v in input_params.items()})
    return input_params


def render_keypoints_2d(dataset, save_dir, overwrite=False):
    """
    render 2d keypoints for each track
    """
    dataset.load_data()
    out_dir = f"{save_dir}/{dataset.seq_name}_joints2d"
    B, T = dataset.n_tracks, dataset.seq_len
    if not overwrite and (os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= B * T):
        print(f"Keypoints already rendered in {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    for i, tid in enumerate(dataset.track_ids):
        joints2d = dataset.data_dict["joints2d"][i]  # (T, J, 3)
        for t, sel_img_name in enumerate(dataset.sel_img_names):
            img = vis_keypoints(joints2d[t : t + 1], dataset.img_size)
            out_path = f"{out_dir}/{sel_img_name}_{tid}.png"
            imageio.imwrite(out_path, img)


def render_results(cfg, dataset, dev_id, res_dicts, out_names, **kwargs):
    """
    render results for all selected phases
    """
    assert len(res_dicts) == len(out_names)
    if len(res_dicts) < 1:
        print("no results to render, skipping")
        return

    B = len(dataset)
    T = dataset.seq_len
    loader = DataLoader(dataset, batch_size=B, shuffle=False)

    device = get_device(dev_id)
    obs_data = move_to(next(iter(loader)), device)
    cam_data = dataset.get_camera_data()

    # load models
    cfg = resolve_cfg_paths(cfg)
    body_model, _ = load_smpl_body_model(cfg.paths.smpl, B * T, device=device)
    vis = init_viewer(
        dataset.img_size,
        cam_data["intrins"][0],
        vis_scale=1.0,
        bg_paths=dataset.sel_img_paths,
        fps=cfg.fps,
    )

    save_paths_all = []
    for res_dict, out_name in zip(res_dicts, out_names):
        res_dict = move_to(res_dict, device)
        scene_dict = prep_result_vis(
            res_dict,
            obs_data["vis_mask"],
            obs_data["track_id"],
            body_model,
        )
        print(kwargs)
        save_paths = animate_scene(
            vis, scene_dict, out_name, seq_name=dataset.seq_name, **kwargs
        )
        save_paths_all.append(save_paths)

    vis.close()

    return save_paths_all


def visualize_log(log_dir, dev_id, phases, save_dir=None, **kwargs):
    print(log_dir)
    cfg = load_config_from_log(log_dir)

    # make sure we get all necessary inputs
    cfg.data.sources = expand_source_paths(cfg.data.sources)
    print("SOURCES", cfg.data.sources)
    # cfg.data.track_ids = "001"
    dataset = get_dataset_from_cfg(cfg)
    if len(dataset) < 1:
        print(f"No tracks in dataset, skipping")
        return

    run_vis(cfg, dataset, log_dir, dev_id, phases=phases, save_dir=save_dir, **kwargs)


def launch_vis(i, args):
    log_dir = args.log_dirs[i]
    dev_id = args.gpus[i % len(args.gpus)]
    os.environ["EGL_DEVICE_ID"] = str(dev_id)
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    path_name = log_dir.split(args.log_root)[-1].strip("/")
    exp_name = "-".join(path_name.split("/")[:2])

    if args.save_root is not None:
        save_dir = f"{args.save_root}/{exp_name}"
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)

    visualize_log(
        log_dir,
        dev_id,
        phases=args.phases,
        save_dir=save_dir,
        overwrite=args.overwrite,
        accumulate=args.accumulate,
        render_kps=args.render_kps,
        render_layers=args.render_layers,
        render_views=args.render_views,
        save_frames=args.save_frames,
        make_grid=args.grid,
    )


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

    if len(args.gpus) > 1:
        from torch.multiprocessing import Pool

        torch.multiprocessing.set_start_method("spawn")

        with Pool(processes=len(args.gpus)) as pool:
            res = pool.starmap(
                launch_vis, [(i, args) for i in range(len(args.log_dirs))]
            )
        return

    for i in range(len(args.log_dirs)):
        launch_vis(i, args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--save_root", default=None)
    parser.add_argument("--phases", nargs="*", default=["motion_chunks"])
    parser.add_argument("--gpus", nargs="*", default=[0])
    parser.add_argument(
        "-rv",
        "--render_views",
        nargs="*",
        default=["src_cam", "front", "above", "side"],
    )
    parser.add_argument("-g", "--grid", action="store_true")
    parser.add_argument("-rl", "--render_layers", action="store_true")
    parser.add_argument("-kp", "--render_kps", action="store_true")
    parser.add_argument("-sf", "--save_frames", action="store_true")
    parser.add_argument("-ra", "--accumulate", action="store_true")
    parser.add_argument("-y", "--overwrite", action="store_true")
    args = parser.parse_args()

    main(args)
