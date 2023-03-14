import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf

from data import get_dataset_from_cfg, expand_source_paths

from slahmr.humor.humor_model import HumorModel
from slahmr.optim.base_scene import BaseSceneModel
from slahmr.optim.moving_scene import MovingSceneModel
from slahmr.optim.optimizers import (
    RootOptimizer,
    SmoothOptimizer,
    SMPLOptimizer,
    MotionOptimizer,
    MotionOptimizerChunks,
)
from slahmr.optim.output import (
    save_track_info,
    save_camera_json,
    save_input_poses,
    save_initial_predictions,
)
from slahmr.vis.render import init_renderer

from slahmr.util.loaders import (
    load_vposer,
    load_state,
    load_gmm,
    load_smpl_body_model,
    resolve_cfg_paths,
)
from slahmr.util.logger import Logger
from slahmr.util.tensor import get_device, move_to

from slahmr.run_vis import run_vis


N_STAGES = 3


def run_opt(cfg, dataset, out_dir, device):
    args = cfg.data
    B = len(dataset)
    T = dataset.seq_len
    loader = DataLoader(dataset, batch_size=B, shuffle=False)

    obs_data = move_to(next(iter(loader)), device)
    cam_data = move_to(dataset.get_camera_data(), device)
    print("OBS DATA", obs_data.keys())
    print("CAM DATA", cam_data.keys())

    # save cameras
    cam_R, cam_t = dataset.cam_data.cam2world()
    intrins = dataset.cam_data.intrins
    save_camera_json(f"cameras.json", cam_R, cam_t, intrins)

    # check whether the cameras are static
    # if static, cannot optimize scale
    cfg.model.opt_scale &= not dataset.cam_data.is_static
    Logger.log(f"OPT SCALE {cfg.model.opt_scale}")

    # loss weights for all stages
    all_loss_weights = cfg.optim.loss_weights
    assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
    stage_loss_weights = [
        {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
    ]
    max_loss_weights = {k: max(wts) for k, wts in all_loss_weights.items()}

    # load models
    cfg = resolve_cfg_paths(cfg)
    paths = cfg.paths
    Logger.log(f"Loading pose prior from {paths.vposer}")
    pose_prior, _ = load_vposer(paths.vposer)
    pose_prior = pose_prior.to(device)

    Logger.log(f"Loading body model from {paths.smpl}")
    body_model, fit_gender = load_smpl_body_model(paths.smpl, B * T, device=device)

    margs = cfg.model
    base_model = BaseSceneModel(
        B, T, body_model, pose_prior, fit_gender=fit_gender, **margs
    )
    base_model.initialize(obs_data, cam_data)
    base_model.to(device)

    # save initial results for later visualization
    save_input_poses(dataset, os.path.join(out_dir, "phalp"), args.seq)
    save_initial_predictions(base_model, os.path.join(out_dir, "init"), args.seq)

    opts = cfg.optim.options
    vis_scale = 0.25
    vis = None
    if opts.vis_every > 0:
        vis = init_renderer(
            dataset.img_size,
            cam_data["intrins"][0],
            device,
            vis_scale=vis_scale,
            bg_paths=dataset.sel_img_paths,
        )
    print("OPTIMIZER OPTIONS:", opts)

    writer = SummaryWriter(out_dir)

    optim = RootOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.root.num_iters, out_dir, vis, writer)

    optim = SMPLOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.smpl.num_iters, out_dir, vis, writer)

    args = cfg.optim.smooth
    optim = SmoothOptimizer(
        base_model, stage_loss_weights, opt_scale=args.opt_scale, **opts
    )
    optim.run(obs_data, args.num_iters, out_dir, vis, writer)

    # now optimize motion model
    Logger.log(f"Loading motion prior from {paths.humor}")
    motion_prior = HumorModel(**cfg.humor)
    load_state(paths.humor, motion_prior, map_location="cpu")
    motion_prior.to(device)
    motion_prior.eval()

    Logger.log(f"Loading GMM motion prior from {paths.init_motion_prior}")
    init_motion_prior = load_gmm(paths.init_motion_prior, device=device)

    model = MovingSceneModel(
        B,
        T,
        body_model,
        pose_prior,
        motion_prior,
        init_motion_prior,
        fit_gender=fit_gender,
        **margs,
    ).to(device)

    # initialize motion model with base model predictions
    base_params = base_model.params.get_dict()
    model.initialize(obs_data, cam_data, base_params, cfg.fps)
    model.to(device)

    if "motion_chunks" in cfg.optim:
        args = cfg.optim.motion_chunks
        optim = MotionOptimizerChunks(model, stage_loss_weights, **args, **opts)
        optim.run(obs_data, optim.num_iters, out_dir, vis, writer)

    if "motion_refine" in cfg.optim:
        args = cfg.optim.motion_refine
        optim = MotionOptimizer(model, stage_loss_weights, **args, **opts)
        optim.run(obs_data, args.num_iters, out_dir, vis, writer)


@hydra.main(version_base=None, config_path="confs", config_name="config.yaml")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)

    out_dir = os.getcwd()
    print("out_dir", out_dir)
    Logger.init(f"{out_dir}/opt_log.txt")

    # make sure we get all necessary inputs
    cfg.data.sources = expand_source_paths(cfg.data.sources)
    print("SOURCES", cfg.data.sources)

    dataset = get_dataset_from_cfg(cfg)
    save_track_info(dataset, out_dir)

    if cfg.run_opt:
        device = get_device(0)
        run_opt(cfg, dataset, out_dir, device)

    if cfg.run_vis:
        run_vis(
            cfg, dataset, out_dir, 0, **cfg.get("vis", dict())
        )


if __name__ == "__main__":
    main()
