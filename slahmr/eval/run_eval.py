import os
import glob
import joblib
import json
import pickle

import pandas as pd
import numpy as np
import torch

import .egobody_utils as eb_util
from .tools import (
    load_body_model,
    load_results_all,
    local_align_joints,
    global_align_joints,
    first_align_joints,
    compute_accel_norm,
    run_smpl,
    JointRegressor,
    EGOBODY_ROOT,
    TDPW_ROOT,
)
from .associate import associate_phalp_track_dirs


def stack_torch(x_list, dim=0):
    return torch.stack(
        [torch.from_numpy(x.astype(np.float32)) for x in x_list], dim=dim
    )


def load_3dpw_params(seq_name, start=0, end=-1):
    seq_file = f"{TDPW_ROOT}/sequenceFiles/test/{seq_name}.pkl"
    with open(seq_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    M = len(data["poses"])
    T = len(data["poses"][0])
    end = T + 1 + end if end < 0 else end
    T = end - start
    trans = stack_torch([x[start:end] for x in data["trans"]])  # (M, T, 3)
    poses = stack_torch([x[start:end] for x in data["poses"]])  # (M, T, 72)
    betas = stack_torch([x[None, :10] for x in data["betas"]]).expand(
        M, T, 10
    )  # (M, T, 10)
    keypts2d = stack_torch([x[start:end] for x in data["poses2d"]])  # (M, T, 3, 18)
    valid_cam = stack_torch(
        [x[start:end] for x in data["campose_valid"]]
    ).bool()  # (M, T)
    valid_kp = (keypts2d.reshape(M, T, -1) > 0).any(dim=-1).bool()  # (M, T)
    valid = valid_cam & valid_kp
    genders = ["male" if x == "m" else "female" for x in data["genders"]]  # (M)
    return {
        "root_orient": poses[..., :3],
        "pose_body": poses[..., 3:],
        "trans": trans,
        "betas": betas,
        "keypts2d": keypts2d,
        "valid": valid,
        "genders": genders,
    }


def load_egobody_params(seq_name, start=0, end=-1):
    """
    returns dict of
    - trans (1, T, 3)
    - root_orient (1, T, 3)
    - pose_body (1, T, 63)
    - betas (1, T, 10)
    - gender (str)
    - keypts2d (1, T, J, 3)
    - valid (1, T)
    """
    smpl_dict = eb_util.load_egobody_smpl_params(seq_name, start=start, end=end)
    kps, valid = eb_util.get_egobody_keypoints(seq_name, start=start, end=end)
    smpl_dict["keypts2d"] = torch.from_numpy(kps.astype(np.float32))[None]
    smpl_dict["valid"] = torch.from_numpy(valid.astype(bool))[None]
    return smpl_dict


def eval_result_dir(
    dset_type, res_dir, out_path, joint_reg, dev_id=0, overwrite=False, debug=False
):
    if os.path.isfile(out_path) and not overwrite:
        print(f"{out_path} already exists, skipping.")
        return

    # get the output metadata
    track_file = f"{res_dir}/track_info.json"
    if not os.path.isfile(track_file):
        print(f"{track_file} does not exist, skipping")
        return

    with open(track_file, "r") as f:
        track_dict = json.load(f)
    start, end = track_dict["meta"]["data_interval"]
    seq_name = os.path.basename(res_dir).split("-")[0]
    print("EVALUATING", res_dir, seq_name, start, end)

    # get the associations from PHALP tracks to GT tracks
    track_info = track_dict["tracks"]
    track_ids = sorted(track_info, key=lambda k: track_info[k]["index"])
    print("TRACK IDS", track_ids)

    if dset_type == "egobody":
        # load the GT params
        gt_params = load_egobody_params(seq_name, start, end)
        phalp_dir = f"{EGOBODY_ROOT}/slahmr/track_preds/{seq_name}"
        img_dir = eb_util.get_egobody_img_dir(seq_name)
    elif dset_type == "3dpw":
        gt_params = load_3dpw_params(seq_name, start, end)
        phalp_dir = f"{TDPW_ROOT}/slahmr/track_gt/{seq_name}"
        img_dir = f"{TDPW_ROOT}/imageFiles/{seq_name}"
    else:
        raise NotImplementedError

    # (M, T) GT track index for each frame and each PHALP track
    match_idcs = associate_phalp_track_dirs(
        phalp_dir,
        img_dir,
        track_ids,
        gt_params["keypts2d"],
        start=start,
        end=end,
        debug=debug,
    )
    # M number of PHALP tracks
    M = len(track_ids)

    # get the GT joints
    G, T = gt_params["pose_body"].shape[:2]
    device = torch.device(f"cuda:{dev_id}")
    gt_joints = []
    for g in range(G):
        body_model = load_body_model(T, "smpl", gt_params["genders"][g], device)
        gt_smpl = run_smpl(
            body_model,
            betas=gt_params["betas"][g].to(device),
            trans=gt_params["trans"][g].to(device),
            root_orient=gt_params["root_orient"][g].to(device),
            pose_body=gt_params["pose_body"][g].to(device),
        )
        gt_joints.append(joint_reg(gt_smpl["vertices"]))  # (T, 15, 3)
    gt_joints = torch.stack(gt_joints, dim=0)
    J, D = gt_joints.shape[-2:]

    # select the correct GT person for each track
    gt_valid = gt_params["valid"]  # (G, T)
    idcs = match_idcs.clone().reshape(M, T, 1, 1).expand(-1, -1, J, D)
    idcs[idcs == -1] = 0  # gather dummy for invalid matches
    gt_match_joints = torch.gather(gt_joints, 0, idcs)
    gt_match_valid = torch.gather(gt_valid, 0, idcs[:, :, 0, 0])
    valid = gt_match_valid & (match_idcs != -1)

    # use the vis_mask to get the correct data subsequence
    vis_mask = torch.tensor(
        [track_info[tid]["vis_mask"] for tid in track_ids]
    )  # (M, T)
    vis_tracks = torch.where(vis_mask.any(dim=1))[0]  # (B,)
    vis_idcs = torch.where(vis_mask.any(dim=0))[0]
    sidx, eidx = vis_idcs.min(), vis_idcs.max() + 1
    L = eidx - sidx

    valid_seq = valid[vis_tracks, sidx:eidx]  # (B, L)
    gt_seq_joints = gt_match_joints[vis_tracks, sidx:eidx]  # (B, L, *)
    gt_seq_joints = gt_seq_joints[valid_seq]

    if debug:
        print(f"vis start {sidx}, end {eidx}, L {L}")
        print("valid track matches", (match_idcs != -1).sum())
        print("filtered gt joints", gt_seq_joints.shape)

    # get the outputs of each phase
    PHASES = ["root_fit", "smooth_fit", "motion_chunks"]
    metric_names = ["ga_jmse", "fa_jmse", "pampjpe", "acc_norm"]
    phase_metrics = {name: [-1 for _ in PHASE] for name in metric_names}
    cur_metrics = {name: np.nan for name in metric_names}
    for i, phase in enumerate(PHASES):
        res_dict = load_results_all(os.path.join(res_dir, phase), device)
        if res_dict is None:
            print(f"PHASE {phase} did not optimize")
            # update all metrics for this phase
            for name in metric_names:
                phase_metrics[name][i] = float(cur_metrics[name])
            print(phase, phase_metrics)
            continue

        # (M, L, -1, 3) verts, (M, L) mask
        res_verts = res_dict["vertices"][valid_seq]
        res_joints = joint_reg(res_verts)  # (*, 15, 3_

        for name in metric_names:
            if name == "acc_norm":
                target = compute_accel_norm(gt_seq_joints)  # (T-2, J)
                pred = compute_accel_norm(res_joints)
            else:
                target = gt_seq_joints
                if name == "pampjpe":
                    pred = local_align_joints(gt_seq_joints, res_joints)
                if name == "ga_jmse":
                    pred = global_align_joints(gt_seq_joints, res_joints)
                if name == "fa_jmse":
                    pred = first_align_joints(gt_seq_joints, res_joints)
                else:
                    raise NotImplementedError
            cur_metrics[name] = torch.linalg.norm(target - pred, dim=-1).mean()
            phase_metrics[name][i] = float(cur_metrics[name])
            print(phase, name, cur_metrics[name])

    df_dict = {"phases": PHASES}
    df_dict.update(phase_metrics)
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(out_path, index=False)
    print(f"saved metrics to {out_path}")


def parse_job_file(args):
    subseq_names = []
    with open(args.job_file, "r") as f:
        for line in f.readlines():
            cmd_args = line.strip().split()
            seq_name, start_str, end_str = cmd_args[:3]
            start = start_str.split("=")[-1]
            end = end_str.split("=")[-1]
            track_name = "longest-2" if args.dset_type == "3dpw" else "all"
            if len(cmd_args) > 3:
                track_name = cmd_args[3].split("=")[-1]
            subseq_names.append(f"{seq_name}-{track_name}-{start}-{end}")
    return subseq_names


def main(args):
    joint_reg = JointRegressor()
    out_root = args.out_root if args.out_root is not None else args.res_root
    os.makedirs(out_root, exist_ok=True)

    subseq_names = parse_job_file(args)
    for subseq in subseq_names:
        res_dir = os.path.join(args.res_root, subseq)
        out_path = os.path.join(out_root, f"{subseq}.txt")
        eval_result_dir(
            args.dset_type,
            res_dir,
            out_path,
            joint_reg,
            overwrite=args.overwrite,
            debug=args.debug,
        )

    metric_paths = glob.glob(f"{out_root}/[!_]*.txt")
    dfs = [pd.read_csv(path) for path in metric_paths]

    merged = pd.concat(dfs).groupby("phase").mean()
    merged.to_csv(f"{out_root}/_final_metrics.txt")
    print(merged)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dset_type",
        required=True,
        choices=["egobody", "3dpw"],
        help="dataset to evaluate on, choices: (3dpw, egobody)",
    )
    parser.add_argument(
        "-i", "--res_root", required=True, help="root directory of outputs to evaluate"
    )
    parser.add_argument(
        "-f",
        "--job_file",
        required=True,
        help="job file specifying the examples to run and evaluate",
    )
    parser.add_argument(
        "-o",
        "--out_root",
        default=None,
        help="directory to save computed metrics, default is res_root",
    )
    parser.add_argument("-y", "--overwrite", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    main(args)
