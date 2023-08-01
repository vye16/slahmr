import os
import glob
import pandas as pd
from copy import deepcopy
import warnings


def update_args(args):
    args = deepcopy(args)
    if args.type == "egobody":
        if args.root is None:
            warnings.warn(
                "args.root flag is set to None, using root directory in source"
            )
            args.root = "/path/to/egobody"
        if args.seqs is None:
            args.seqs = get_egobody_seqs(args.root, args.split)
        args.img_dirs = [
            glob.glob(f"{args.root}/egocentric_color/{seq}/**/PV")[0]
            for seq in args.seqs
        ]
        return args

    if args.type == "3dpw":
        if args.root is None:
            warnings.warn(
                "args.root flag is set to None, using root directory in source"
            )
            args.root = "/path/to/3DPW"
        if args.split == "val":
            args.split = "validation"
        if args.seqs is None:
            args.seqs = get_3dpw_seqs(args.root, args.split)
        args.img_name = "imageFiles"

    elif args.type == "posetrack":
        if args.root is None:
            warnings.warn(
                "args.root flag is set to None, using root directory in source"
            )
            args.root = "/path/to/posetrack/posetrack2018/posetrack_data"
        args.img_name = f"images/{args.split}"

    elif args.type == "davis":
        if args.root is None:
            warnings.warn(
                "args.root flag is set to None, using root directory in source"
            )
            args.root = "/path/to/DAVIS"
        args.img_name = "JPEGImages/Full-Resolution"
        args.split = ""

    else:
        assert args.root is not None
        args.img_name = args.img_name if args.img_name is not None else "images"
        args.split = ""

    args.img_dirs, args.seqs = get_all_img_dirs(
        f"{args.root}/{args.img_name}", args.seqs
    )
    return args


def isimage(f):
    ext = os.path.splitext(f)[-1].lower()
    return ext in [".png", ".jpg"]


def get_all_img_dirs(data_root, seqs=None):
    """
    returns all image directories in root, recursively searching, and the sequence names
    """
    seqs = seqs if seqs is not None else []
    img_dirs = []

    def get_name(p):
        return p.removeprefix(f"{data_root}/")

    for root, _, files in os.walk(data_root):
        num_imgs = len([f for f in files if isimage(f)])
        if num_imgs > 0:
            name = get_name(root)
            if len(seqs) == 0 or name in seqs:
                img_dirs.append(root)
    img_dirs = sorted(img_dirs)
    seqs = [get_name(d) for d in img_dirs]
    return img_dirs, seqs


def get_custom_seqs(data_root):
    img_dir = f"{data_root}/images"
    if not os.path.isdir(img_dir):
        return []
    return sorted(os.listdir(img_dir))


def get_egobody_seqs(data_root, split):
    split_file = f"{data_root}/data_splits.csv"
    df = pd.read_csv(split_file)
    if split not in df.columns:
        print(f"{split} not in {split_file}")
        return []
    return sorted(df[split].dropna().tolist())


def get_3dpw_seqs(data_root, split):
    split_dir = f"{data_root}/sequenceFiles/{split}"
    if not os.path.isdir(split_dir):
        return []
    seq_files = sorted(os.listdir(split_dir))
    return [os.path.splitext(f)[0] for f in seq_files]


def get_posetrack_seqs(data_root, split):
    split_dir = f"{data_root}/images/{split}"
    if not os.path.isdir(split_dir):
        return []
    return sorted(os.listdir(split_dir))


def get_davis_seqs(data_root):
    img_root = f"{data_root}/JPEGImages/Full-Resolution"
    if not os.path.isdir(img_root):
        return []
    return sorted(os.listdir(img_root))


def get_img_dir(data_type, data_root, seq, img_name):
    if data_type == "posetrack":
        return f"{data_root}/{img_name}/{seq}"
    if data_type == "3dpw":
        return f"{data_root}/imageFiles/{seq}"
    if data_type == "davis":
        return f"{data_root}/JPEGImages/Full-Resolution/{seq}"
    if data_type == "custom":
        return f"{data_root}/images/{seq}"  # custom sequence
    return None
