import os
import glob
import pickle
import numpy as np
import json


def export_cameras(seq_data, cam_dir):
    os.makedirs(cam_dir, exist_ok=True)
    K = seq_data["cam_intrinsics"]  # (3, 3)
    # fx, fy, cx, cy
    intrins = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out_path = os.path.join(cam_dir, "intrinsics.txt")
    np.savetxt(out_path, intrins)
    return out_path


def export_keypoints(seq_data, img_dir, kp_dir):
    kps_all = seq_data["poses2d"]  # M list of (T, 3, 18)
    M = len(kps_all)
    T = len(kps_all[0])
    img_files = sorted(glob.glob(f"{img_dir}/*.jpg"))
    img_names = [os.path.splitext(os.path.basename(x))[0] for x in img_files]
    assert T == len(img_names), f"{T}, {len(img_names)}"
    os.makedirs(kp_dir, exist_ok=True)
    for t, name in enumerate(img_names):
        people = []
        for m in range(M):
            kps = kps_all[m][t].T  # (18, 3)
            people.append({"pose_keypoints_2d": kps.tolist()})
        out_dict = {"people": people}

        kp_path = f"{kp_dir}/{name}_keypoints.json"
        with open(kp_path, "w") as f:
            json.dump(out_dict, f, indent=1)
    return kp_dir


def export_seq(data_root, split, seq_name, out_root):
    print(f"Exporting sequence {seq_name}")
    data_path = f"{data_root}/sequenceFiles/{split}/{seq_name}.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    cam_dir = f"{out_root}/cameras_gt/{seq_name}"
    int_path = export_cameras(data, cam_dir)

    kp_dir = f"{out_root}/keypoints_gt/{seq_name}"
    img_dir = f"{data_root}/imageFiles/{seq_name}"
    kp_dir = export_keypoints(data, img_dir, kp_dir)

    return int_path, _, kp_dir


def main(args):
    if not (args.cameras or args.keypoints):
        print("Please specify if you want to export cameras or keypoints")
        return

    src_dir = f"{args.data_root}/sequenceFiles/{args.split}"
    out_root = f"{args.data_root}/slahmr"
    data_files = sorted(glob.glob(f"{src_dir}/*.pkl"))
    print(f"FOUND {len(data_files)} data files in {src_dir}")

    for path in data_files:
        seq_name = os.path.splitext(os.path.basename(path))[0]
        export_seq(args.data_root, args.split, seq_name, out_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/path/to/3DPW")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    main(args)
