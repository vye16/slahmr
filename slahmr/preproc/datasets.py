import os
import glob
import pandas as pd
from copy import deepcopy


def update_args(args):
    args = deepcopy(args)
    if args.type == "custom":
        if args.seqs is None:
            args.seqs = get_custom_seqs(args.root)
        return args
    if args.type == "egobody":
        if args.root is None:
            args.root = "/path/to/egobody"
        if args.seqs is None:
            args.seqs = get_egobody_seqs(args.root, args.split)
        return args
    if args.type == "3dpw":
        if args.root is None:
            args.root = "/path/to/3DPW"
        if args.seqs is None:
            args.seqs = get_3dpw_seqs(args.root, args.split)
        return args
    elif args.type == "posetrack":
        if args.root is None:
            args.root = "/path/to/posetrack/posetrack2018/posetrack_data"
        if args.seqs is None:
            args.seqs = get_posetrack_seqs(args.root, args.split)
        return args
    elif args.type == "davis":
        if args.root is None:
            args.root = "/path/to/DAVIS"
            if args.seqs is None:
                args.seqs = get_davis_seqs(args.root)
        return args
    raise NotImplementedError


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


def get_img_dir(data_type, data_root, seq, split):
    if data_type == "posetrack":
        return f"{data_root}/images/{split}/{seq}"
    if data_type == "egobody":
        return glob.glob(f"{data_root}/egocentric_color/{seq}/**/PV")[0]
    if data_type == "3dpw":
        return f"{data_root}/imageFiles/{seq}"
    if data_type == "davis":
        return f"{data_root}/JPEGImages/Full-Resolution/{seq}"
    return f"{data_root}/images/{seq}"  # custom sequence
