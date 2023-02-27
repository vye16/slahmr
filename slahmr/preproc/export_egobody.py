import os
import glob
import numpy as np


def read_camera_params(path):
    out_dict = {}
    with open(path, "r") as f:
        header = np.fromstring(f.readline().strip(), sep=",")
        for row in f.readlines():
            tokens = row.strip().split(",")
            out_dict[tokens[0]] = [float(x) for x in tokens[1:]]
    return out_dict, header


def export_cameras(img_dir, cam_dir):
    cam_path = glob.glob(f"{img_dir}/*pv.txt")
    assert len(cam_path) == 1, f"no camera path found in {img_dir}"
    cam_path = cam_path[0]
    cam_dict, cam_meta = read_camera_params(cam_path)

    cx, cy, w, h = cam_meta

    extrins = []
    intrins = []

    img_files = sorted(glob.glob(f"{img_dir}/PV/*.jpg"))
    for img_file in img_files:
        # get the timestamps of each image file and get corresponding parameters
        ts = os.path.basename(img_file).split("_")[0]
        data = cam_dict[ts]
        fx, fy = data[:2]
        c2w = data[2:]
        assert len(c2w) == 16

        intrins.append([fx, fy, cx, cy, w, h])
        extrins.append(c2w)

    intrins = np.array(intrins)
    extrins = np.array(extrins)

    os.makedirs(cam_dir, exist_ok=True)
    int_path = os.path.join(cam_dir, "intrinsics.txt")
    ext_path = os.path.join(cam_dir, "cam2world.txt")
    np.savetxt(int_path, intrins)
    np.savetxt(ext_path, extrins.reshape(-1, 16))
    print(f"intrinsics and extrinsics saved to {cam_dir}")
    return int_path, ext_path


def export_keypoints(img_dir, out_path):
    kp_file = f"{img_dir}/keypoints.npz"
    valid_file = f"{img_dir}/valid_frame.npz"
    img_paths = sorted(glob.glob(f"{img_dir}/PV/*.jpg"))
    img_names = [os.path.basename(x) for x in img_paths]

    kp_dict = {}
    valid_dict = {}
    kp_data = np.load(kp_file)
    valid_data = np.load(valid_file)

    zeros = np.zeros_like(kp_data["keypoints"][0])
    print(zeros.shape)
    for img_path, kps in zip(kp_data["imgname"], kp_data["keypoints"]):
        img_name = os.path.basename(img_path)
        kp_dict[img_name] = kps

    for img_path, valid in zip(valid_data["imgname"], valid_data["valid"]):
        img_name = os.path.basename(img_path)
        valid_dict[img_name] = valid

    kps = np.stack([kp_dict.get(name, zeros) for name in img_names], axis=0)
    valid = np.stack([valid_dict.get(name, False) for name in img_names], axis=0)
    np.savez(out_path, img_names=img_names, keypoints=kps, valid=valid)
    print(f"keypoints saved to {out_path}")


def export_seq(img_dir, out_root):
    kp_root = f"{out_root}/keypoints_gt"
    cam_root = f"{out_root}/cameras_gt"
    os.makedirs(kp_root, exist_ok=True)
    os.makedirs(cam_root, exist_ok=True)

    seq_name = img_dir.strip("/").split("/")[-2]
    kp_path = f"{kp_root}/{seq_name}.npz"
    cam_dir = f"{cam_root}/{seq_name}"
    export_keypoints(img_dir, kp_path)
    int_path, ext_path = export_cameras(img_dir, cam_dir)
    return int_path, ext_path, kp_path


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/path/to/egobody")
    args = parser.parse_args()


    img_root = f"{args.data_root}/egocentric_color"
    img_dirs = glob.glob(f"{img_root}/**/**")
    out_root = f"{args.data_root}/slahmr"

    for img_dir in img_dirs:
        export_seq(img_dir, out_root)
