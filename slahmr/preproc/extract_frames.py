import argparse
import imageio
import os
import subprocess


def video_to_frames(
    path,
    out_dir,
    fps=30,
    ext="jpg",
    down_scale=1,
    start_sec=0,
    end_sec=-1,
    overwrite=False,
    **kwargs,
):
    """
    :param path
    :param out_dir
    :param fps
    :param down_scale (optional int)
    """
    os.makedirs(out_dir, exist_ok=True)

    arg_str = f"-copyts -vf fps={fps}"
    if down_scale != 1:
        arg_str = f"{arg_str},scale='iw/{down_scale}:ih/{down_scale}'"
    if start_sec > 0:
        arg_str = f"{arg_str} -ss {start_sec}"
    if end_sec > start_sec:
        arg_str = f"{arg_str} -to {end_sec}"

    yn = "-y" if overwrite else "-n"
    cmd = f"ffmpeg -i {path} {arg_str} {out_dir}/%06d.{ext} {yn}"
    print(cmd)

    return subprocess.call(cmd, shell=True, stdin=subprocess.PIPE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="path to video")
    parser.add_argument(
        "--out_root", type=str, required=True, help="output dir for frames"
    )
    parser.add_argument(
        "--seqs",
        nargs="*",
        default=None,
        help="[optional] sequences to run, default runs all available",
    )
    parser.add_argument("--fps", type=int, default=30, help="fps to extract frames")
    parser.add_argument(
        "--ext", type=str, default="jpg", help="output filetype for frames"
    )
    parser.add_argument(
        "--down_scale", type=int, default=1, help="scale to extract frames"
    )
    parser.add_argument(
        "-ss", "--start_sec", type=float, default=0, help="seconds to start_sec"
    )
    parser.add_argument(
        "-es", "--end_sec", type=float, default=-1, help="seconds to end_sec"
    )
    parser.add_argument(
        "-y", "--overwrite", action="store_true", help="overwrite if already exist"
    )
    args = parser.parse_args()
    seqs_all = os.listdir(args.data_root)
    if args.seqs is None:
        args.seqs = seqs_all

    for seq in args.seqs:
        path = os.path.join(args.data_root, seq)
        print(f"EXTRACTING FRAMES FROM {path}")
        assert os.path.isfile(path)
        seq_name = os.path.splitext(os.path.basename(path.rstrip("/")))[0]
        out_dir = os.path.join(args.out_root, seq_name)
        video_to_frames(
            path,
            out_dir,
            args.fps,
            args.ext,
            args.down_scale,
            args.start_sec,
            args.end_sec,
            args.overwrite,
        )
