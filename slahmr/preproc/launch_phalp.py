import os
import subprocess
import multiprocessing as mp
from concurrent import futures

from preproc.datasets import update_args
from preproc.export_phalp import export_sequence_results


def launch_phalp(gpus, seq, img_dir, res_dir, overwrite=False):
    """
    run phalp using GPU pool
    """
    cur_proc = mp.current_process()
    print("PROCESS", cur_proc.name, cur_proc._identity)
    # 1-indexed processes
    worker_id = cur_proc._identity[0] - 1 if len(cur_proc._identity) > 0 else 0
    gpu = gpus[worker_id % len(gpus)]

    PHALP_DIR = os.path.abspath(f"{__file__}/../")
    print("PHALP DIR", PHALP_DIR)

    cmd_args = [
        f"cd {PHALP_DIR};",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "python track.py",
        f"video.source={img_dir}",
        f"video.output_dir={res_dir}",
        f"overwrite={overwrite}",
        "detect_shots=True",
        "video.extract_video=False",
        "render.enable=False",
    ]

    cmd = " ".join(cmd_args)
    print(cmd)
    return subprocess.call(cmd, shell=True)


def process_seq(
    gpus,
    out_root,
    seq,
    img_dir,
    out_name="phalp_out",
    track_name="track_preds",
    shot_name="shot_idcs",
    overwrite=False,
):
    """
    Run and export PHALP results
    """
    name = os.path.basename(seq)
    res_root = f"{out_root}/{out_name}/{seq}"
    os.makedirs(res_root, exist_ok=True)
    res_dir = os.path.join(res_root, "results")
    res_path = f"{res_root}/{name}.pkl"
    if overwrite or not os.path.isfile(res_path):
        res = launch_phalp(gpus, seq, img_dir, res_root, overwrite)
        os.rename(f"{res_dir}/demo_{name}.pkl", res_path)
        assert res == 0, "PHALP FAILED"

    # export the PHALP predictions
    track_dir = f"{out_root}/{track_name}/{seq}"
    shot_path = f"{out_root}/{shot_name}/{seq}.json"

    export_sequence_results(res_path, track_dir, shot_path)
    return 0


def get_out_dir(src_root, src_dir, src_token, out_token):
    """
    :param src_root (str) root of all data
    :param src_dir (str) img input dir
    :param src_token (str) parent name of image input dir
    :param out_token (str) name of output dir
    """
    src_suffix = src_dir.removeprefix(src_root)
    out_dir = f"{out_root}/{src_suffix}"
    return out_dir.replace(src_token, out_token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="posetrack", help="dataset to process")
    parser.add_argument("--root", default=None, help="root dir of data, default None")
    parser.add_argument("--split", default="val", help="split of dataset, default val")
    parser.add_argument(
        "--img_name", default=None, help="input image directory name, default None"
    )
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--gpus", nargs="*", default=[0])
    parser.add_argument("-y", "--overwrite", action="store_true")

    args = parser.parse_args()
    args = update_args(args)

    out_root = f"{args.root}/slahmr/{args.split}"

    print(f"running phalp on {len(args.img_dirs)} image directories")
    if len(args.gpus) > 1:
        with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as exe:
            for img_dir, seq in zip(args.img_dirs, args.seqs):
                exe.submit(
                    process_seq,
                    args.gpus,
                    out_root,
                    seq,
                    img_dir,
                    overwrite=args.overwrite,
                )
    else:
        for img_dir, seq in zip(args.img_dirs, args.seqs):
            process_seq(args.gpus, out_root, seq, img_dir, overwrite=args.overwrite)
