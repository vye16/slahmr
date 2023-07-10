import os
import subprocess
import multiprocessing as mp
from concurrent import futures

from preproc.datasets import update_args, get_img_dir
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

    base_path, sample = img_dir.split(seq)[:2]
    cmd_args = [
        f"cd {PHALP_DIR};",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "python track.py",
        f"video.source={img_dir}",
        f"video.output_dir={res_dir}",
        f"overwrite={overwrite}",
        "detect_shots=True",
        "video.extract_video=False",
    ]

    cmd = " ".join(cmd_args)
    print(cmd)
    return subprocess.call(cmd, shell=True)


def process_seq(
    gpus,
    seq,
    img_dir,
    res_dir,
    track_name="track_preds",
    shot_name="shot_idcs",
    overwrite=False,
):
    """
    Run and export PHALP results
    """
    res_path = f"{res_dir}/results/{seq}.pkl"
    if overwrite or not os.path.isfile(res_path):
        res = launch_phalp(gpus, seq, img_dir, res_dir, overwrite)
        os.rename(f"{res_dir}/results/demo_{seq}.pkl", res_path)
        assert res == 0, "PHALP FAILED"

    # export the PHALP predictions
    out_root, out_name = os.path.split(res_dir)
    export_sequence_results(
        out_root,
        seq,
        res_name=f"{out_name}/results",
        track_name=track_name,
        shot_name=shot_name,
    )
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="posetrack", help="dataset to process")
    parser.add_argument("--root", default=None, help="root dir of data, default None")
    parser.add_argument("--split", default="val", help="split of dataset, default val")
    parser.add_argument(
        "--out_name", default="phalp_out", help="output name, default phalp_out"
    )
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--gpus", nargs="*", default=[0])
    parser.add_argument("-y", "--overwrite", action="store_true")

    args = parser.parse_args()
    args = update_args(args)

    print(f"running phalp on {len(args.seqs)} sequences")
    if len(args.gpus) > 1:
        with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as exe:
            for seq in args.seqs:
                img_dir = get_img_dir(args.type, args.root, seq, args.split)
                res_dir = f"{args.root}/slahmr/{args.out_name}"
                exe.submit(
                    process_seq,
                    args.gpus,
                    seq,
                    img_dir,
                    res_dir,
                    overwrite=args.overwrite,
                )
    else:
        for seq in args.seqs:
            img_dir = get_img_dir(args.type, args.root, seq, args.split)
            res_dir = f"{args.root}/slahmr/{args.out_name}"
            process_seq(args.gpus, seq, img_dir, res_dir, overwrite=args.overwrite)
