import os

ROOT_DIR = os.path.abspath(f"{__file__}/../../../")
SRC_DIR = os.path.join(ROOT_DIR, "third-party/DROID-SLAM")
print("PROJ SRC", ROOT_DIR)
print("DROID SRC", SRC_DIR)

import sys

sys.path.append(SRC_DIR)
sys.path.append(f"{SRC_DIR}/droid_slam")

import glob
import shutil
import argparse
from tqdm import tqdm

import json

import cv2
import trimesh
import numpy as np
import torch

from lietorch import SE3
from droid import Droid
import droid_backends


def get_image(image):
    return image.permute(1, 2, 0).cpu().numpy()[..., ::-1]


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("image", image / 255.0)
    cv2.waitKey(1)


def get_hwf(test_path):
    test_img = cv2.imread(test_path)
    H, W, _ = test_img.shape
    F = 0.5 * (H + W)
    return H, W, F


def isimage(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext == ".png" or ext == ".jpg" or ext == ".jpeg"


def get_image_files(img_dir, stride, start=0, end=-1):
    image_list = sorted(list(filter(isimage, os.listdir(img_dir))))
    end = len(image_list) + 1 + end if end < 0 else end
    return [os.path.join(img_dir, name) for name in image_list[start:end:stride]]


def load_intrins(intrins_path, image_files, start=0, end=-1):
    N = len(image_files)
    H, W, F = get_hwf(image_files[0])
    # intrinsics for default FOV
    default = torch.tensor([F, F, W / 2, H / 2, W, H])[None].repeat(N, 1)
    if intrins_path is None:
        return default
    assert os.path.isfile(intrins_path), f"{intrins_path} DOES NOT EXIST"
    assert intrins_path.endswith(".txt"), f"{intrins_path} INCORRECT EXT"

    print(f"LOADING INTRINSICS FROM {intrins_path}")
    intrins = torch.from_numpy(np.loadtxt(intrins_path).astype(np.float32))
    print(intrins.shape, N)
    if intrins.ndim < 2:  # assume intrins is same for all frames
        intrins = intrins[None].repeat(N, 1)
    else:
        end = N + 1 + end if end < 0 else end
        intrins = intrins[start:end]
    print(intrins.shape)
    if intrins.shape[1] == 4:  # fx, fy, cx, cy, adding on W and H
        ones = torch.ones_like(intrins[:, :1])
        intrins = torch.cat([intrins, ones * W, ones * H], dim=-1)
    return intrins


def image_stream(image_files, intrins_all):
    """image generator"""

    N = len(image_files)

    assert intrins_all.shape[0] == N
    assert intrins_all.shape[1] >= 4

    for t, imfile in enumerate(image_files):
        image = cv2.imread(imfile)
        if image is None:
            print(imfile, "is none, exiting")
            sys.exit(1)

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        fx, fy, cx, cy = intrins_all[t, :4]
        sx, sy = w1 / w0, h1 / h0
        frame_intrins = torch.as_tensor([fx * sx, fy * sy, cx * sx, cy * sy])
        yield t, image[None], frame_intrins


def unpack_video(video, depth_filter_thresh=0.005):
    """
    extract the video of keyframe disps and poses
    returns:
        (T) timestamps
        (T, 7) world2cam pose for each keyframe
        (T, H, W) disps
        (T, H, W, 3) images
        (T, H, W) mask of valid pixels
        (4) intrinsics (fx, fy, cx, cy)
    """
    with torch.no_grad():
        with video.get_lock():
            # dirty frames have disparity normalized over all previous frames
            # should probably change this in visualizer
            t = video.counter.value

        print(f"{t} keyframes in map")

        # normalization happens after all frames have been processed
        if t < 1:
            print("no keyframes")
            return

        # taken from visualization.py#80
        tstamps = video.tstamp[:t]
        dirty_index = torch.arange(t, device=video.poses.device)

        poses = torch.index_select(video.poses, 0, dirty_index)  # (T, 7)
        disps = torch.index_select(video.disps, 0, dirty_index)  # (T, H, W)
        images = torch.index_select(video.images, 0, dirty_index)  # (T, 3, H, W)

        images = images[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
        intrins = video.intrinsics[0]  # (4,)

        # get mask of static points
        thresh = depth_filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))
        count = droid_backends.depth_filter(
            video.poses, video.disps, intrins, dirty_index, thresh
        )
        disp_mean = disps.mean(dim=(1, 2), keepdim=True).clamp_min(0.2)
        valid = (count >= 2) & (disps > 0.5 * disp_mean)  # (T, H, W)
        return {
            "tstamps": tstamps.cpu(),
            "w2c": poses.cpu(),
            "disps": disps.cpu(),
            "images": images.cpu(),
            "valid": valid.cpu(),
            "intrins": intrins.cpu(),
        }


def get_keyframe_map(video_dict):
    """
    :param video_dict (dict) unpacked droid slam video
    :returns
        (T, 4, 4) camera to world matrix
        (N, 3) valid points in world
        (N, 3) colors of valid points
    """
    poses = video_dict["w2c"]
    images = video_dict["images"]
    valid = video_dict["valid"]
    t = len(poses)
    c2w = SE3(poses).inv()

    # `iproj` only has a cuda backend; if we pass in CPU tensors, it will silently
    # return zeros
    points = droid_backends.iproj(
        c2w.data.cuda(),
        video_dict["disps"].cuda(),
        video_dict["intrins"].cuda(),
    )
    valid_pts = torch.cat([points[i, valid[i]] for i in range(t)], dim=0)
    valid_rgb = torch.cat([images[i, valid[i]] for i in range(t)], dim=0)
    return (
        c2w.matrix().detach().cpu(),
        valid_pts.detach().cpu(),
        valid_rgb.detach().cpu(),
    )


def get_frame_cameras(droid, img_paths, intrins_all):
    N = len(img_paths)

    t = droid.video.counter.value
    print(f"{t} keyframes in map")

    if t > 1:
        with torch.no_grad():
            # localize all frames and get edges into keyframe graph
            # returns 7D tensor (3D trans, 4D quat)
            c2w = droid.terminate(image_stream(img_paths, intrins_all))
        c2w = torch.from_numpy(c2w.astype(np.float32))
        return SE3(c2w).inv().matrix()

    return torch.eye(4)[None].repeat(N, 1, 1)


def save_camera_json(path, extrins, intrins, yup=False):
    """
    :param path
    :param extrins (N, 4, 4)
    :param intrins (N, 4)
    """
    N = extrins.shape[0]
    if yup:
        T = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        extrins = torch.matmul(T[None], extrins)
    with open(path, "w") as f:
        json.dump(
            {
                "rotation": extrins[:, :3, :3].reshape((N, 9)).tolist(),
                "translation": extrins[:, :3, 3].tolist(),
                "intrinsics": intrins.tolist(),
            },
            f,
            indent=1,
        )


def save_pcl(path, points, colors, yup=False):
    assert len(points) == len(colors)
    if yup:
        T = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        points = torch.einsum("ij,...j->...i", T, points)  # (*, 3)
    points = points.reshape(-1, 3).numpy()
    colors = colors.reshape(-1, 3).numpy()
    obj = trimesh.Trimesh(vertices=points, vertex_colors=colors)
    with open(path, "wb") as f:
        f.write(trimesh.exchange.ply.export_ply(obj))


def save_keyframe_map(out_dir, kf_nodes):
    """
    Saves everything with x right y up z back
    (converts from x right y down z forward)
    :param out_dir
    :param kf_nodes (dict)
    """
    c2w, points, colors = get_keyframe_map(kf_nodes)
    intrins = kf_nodes["intrins"].detach().cpu()

    print("c2w", c2w.shape)
    print("points", points.shape)
    print("colors", colors.shape)

    os.makedirs(out_dir, exist_ok=True)
    save_pcl(f"{out_dir}/map_points.ply", points, colors)
    save_camera_json(f"{out_dir}/map_cameras.json", c2w, intrins)
    print(f"saved {len(c2w)} keyframes to {out_dir}")


def save_cameras(map_dir, frame_w2c, intrins):
    os.makedirs(map_dir, exist_ok=True)
    print("SAVING CAMERA INTRINS AND EXTRINS IN", map_dir)

    # save all camera extrinsics and intrinsic
    W, H = intrins[0, 4], intrins[0, 5]
    focal = intrins[:, :2].mean()
    print("INTRINS", intrins[0])
    np.savez(
        f"{map_dir}/cameras.npz",
        height=H,
        width=W,
        focal=focal,
        intrins=intrins[:, :4],
        w2c=frame_w2c,
    )

    # save visualization
    frame_intrins = torch.tensor([focal, focal, W / 2, H / 2, W, H])
    frame_c2w = torch.linalg.inv(frame_w2c)
    save_camera_json(f"{map_dir}/frame_cameras.json", frame_c2w, intrins)


def main(args):
    args.stereo = False
    torch.multiprocessing.set_start_method("spawn")

    droid = None

    img_paths = get_image_files(
        args.img_dir, args.stride, start=args.start, end=args.end
    )
    if args.map_dir is not None:
        os.makedirs(args.map_dir, exist_ok=True)
        camera_file = os.path.join(args.map_dir, "cameras.npz")
        if os.path.isfile(camera_file) and not args.overwrite:
            print(glob.glob(f"{args.map_dir}/*.npz"), "already exist, skipping")
            return

    intrins_all = load_intrins(
        args.intrins_path, img_paths, start=args.start, end=args.end
    )
    print("intrins shape", intrins_all.shape, "num images", len(img_paths))

    for t, image, intrinsics in tqdm(image_stream(img_paths, intrins_all)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)

        droid.track(t, image, intrinsics=intrinsics)

    if args.map_dir is None:
        return

    # save cameras
    frame_w2c = get_frame_cameras(droid, img_paths, intrins_all)
    save_cameras(args.map_dir, frame_w2c, intrins_all)

    # save keyframe cameras and points
    kf_nodes = unpack_video(droid.video)
    save_keyframe_map(args.map_dir, kf_nodes)


def get_slam_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", default=1, type=int, help="frame stride")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--weights", default=f"{ROOT_DIR}/_DATA/droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="weight for translation / rotation components of flow",
    )
    parser.add_argument(
        "--filter_thresh",
        type=float,
        default=1.0,
        help="how much motion before considering new keyframe",
    )
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument(
        "--keyframe_thresh",
        type=float,
        default=4.0,
        help="threshold to create a new keyframe",
    )
    parser.add_argument(
        "--frontend_thresh",
        type=float,
        default=16.0,
        help="add edges between frames whithin this distance",
    )
    parser.add_argument(
        "--frontend_window", type=int, default=25, help="frontend optimization window"
    )
    parser.add_argument(
        "--frontend_radius",
        type=int,
        default=2,
        help="force edges between frames within radius",
    )
    parser.add_argument(
        "--frontend_nms", type=int, default=1, help="non-maximal supression of edges"
    )

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    return parser


if __name__ == "__main__":
    parser = get_slam_parser()
    parser.add_argument(
        "-i", "--img_dir", required=True, type=str, help="path to image directory"
    )
    parser.add_argument(
        "-o", "--map_dir", type=str, default=None, help="path to output directory"
    )
    parser.add_argument(
        "-y", "--overwrite", action="store_true", help="overwrite existing cameras"
    )
    parser.add_argument(
        "-g", "--save_graph", action="store_true", help="save the frame graph"
    )
    parser.add_argument(
        "--intrins_path", default=None, help="path to camera intrinsics"
    )
    parser.add_argument("--start", default=0, type=int, help="start frame")
    parser.add_argument("--end", default=-1, type=int, help="end frame")

    args = parser.parse_args()
    main(args)
