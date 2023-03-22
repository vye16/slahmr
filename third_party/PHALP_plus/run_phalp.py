import argparse
import os
import time
import traceback
import warnings

import cv2
import imageio
import joblib
import numpy as np
from deep_sort_ import nn_matching
from deep_sort_.detection import Detection
from deep_sort_.tracker import Tracker
from models.pose_model import PoseModel
from PHALP import PHALP_tracker
from pytube import YouTube
from tqdm import tqdm
from utils.make_video import render_frame_main_online
from utils.utils import FrameExtractor, str2bool

warnings.filterwarnings("ignore")


def test_tracker(opt, phalp_tracker: PHALP_tracker):
    print("running PHALP on", opt.video_seq)
    print("base_path", opt.base_path)
    print("storage_folder", opt.storage_folder)

    eval_keys = ["tracked_ids", "tracked_bbox", "tid", "bbox", "tracked_time"]
    history_keys = ["appe", "loca", "pose", "uv"] if opt.render else []
    prediction_keys = (
        ["prediction_uv", "prediction_pose", "prediction_loca"] if opt.render else []
    )
    extra_keys_1 = [
        "center",
        "scale",
        "size",
        "img_path",
        "img_name",
        "mask_name",
        "conf",
    ]
    extra_keys_2 = ["smpl", "3d_joints", "camera", "embedding", "vitpose"]
    history_keys = history_keys + extra_keys_1 + extra_keys_2
    visual_store_ = eval_keys + history_keys + prediction_keys
    tmp_keys_ = ["uv", "prediction_uv", "prediction_pose", "prediction_loca"]

    res_dir = os.path.join(opt.storage_folder, "results")
    res_file = f"{res_dir}/{opt.video_seq}.pkl"
    if not (opt.overwrite) and os.path.isfile(res_file):
        print(f"{res_file} already exists, skipping")
        return 0

    print("WRITING RESULTS TO", res_file)

    mask_dir = os.path.join(opt.storage_folder, "_TMP")
    vis_dir = os.path.join(opt.storage_folder, "vis_render")

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    phalp_tracker.eval()
    phalp_tracker.HMAR.reset_nmr(opt.res)

    # initialize vitpose model
    device = "cuda"
    cpm = PoseModel(device)

    metric = nn_matching.NearestNeighborDistanceMetric(
        opt, opt.hungarian_th, opt.past_lookback
    )
    tracker = Tracker(
        opt,
        metric,
        max_age=opt.max_age_track,
        n_init=opt.n_init,
        phalp_tracker=phalp_tracker,
        dims=[4096, 4096, 99],
    )

    try:

        main_path_to_frames = opt.base_path + "/" + opt.video_seq + opt.sample
        list_of_frames = np.sort(
            [i for i in os.listdir(main_path_to_frames) if ".jpg" in i]
        )
        list_of_frames = (
            list_of_frames
            if opt.start_frame == -1
            else list_of_frames[opt.start_frame : opt.end_frame]
        )
        list_of_shots = phalp_tracker.get_list_of_shots(
            main_path_to_frames, list_of_frames
        )

        tracked_frames = []
        final_visuals_dic = {}

        for t_, frame_name in enumerate(tqdm(list_of_frames)):
            if opt.verbose:
                print(
                    "\n\n\nTime: ",
                    opt.video_seq,
                    frame_name,
                    t_,
                    time.time() - time_ if t_ > 0 else 0,
                )
                time_ = time.time()

            image_frame = cv2.imread(main_path_to_frames + "/" + frame_name)
            img_height, img_width, _ = image_frame.shape
            new_image_size = max(img_height, img_width)
            top, left = (
                (new_image_size - img_height) // 2,
                (new_image_size - img_width) // 2,
            )
            measurments = [img_height, img_width, new_image_size, left, top]
            opt.shot = 1 if t_ in list_of_shots else 0

            ############ detection ##############
            (
                pred_bbox,
                pred_masks,
                pred_scores,
                mask_names,
                gt,
            ) = phalp_tracker.get_detections(image_frame, frame_name, t_)

            ############ vitpose #############
            vitposes_out = cpm.predict_pose(
                image_frame[:, :, ::-1],
                [np.concatenate([pred_bbox, pred_scores[:, None]], axis=1)],
            )
            # vis = cpm.visualize_pose_results(image_frame[:,:,::-1], vitposes_out)
            # cv2.imwrite('test.jpg', vis[:,:,::-1])

            ############ HMAR ##############
            detections = []
            for bbox, mask, score, mask_name, gt_id, vitpose_out in zip(
                pred_bbox, pred_masks, pred_scores, mask_names, gt, vitposes_out
            ):
                if bbox[2] - bbox[0] < 50 or bbox[3] - bbox[1] < 100:
                    continue
                detection_data = phalp_tracker.get_human_apl(
                    image_frame,
                    mask,
                    bbox,
                    score,
                    [main_path_to_frames, frame_name],
                    mask_name,
                    t_,
                    measurments,
                    gt_id,
                )
                vitpose_2d = np.zeros([25, 3])
                vitpose_2d[
                    [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
                ] = vitpose_out["keypoints"]
                detection_data["vitpose"] = vitpose_2d
                detections.append(Detection(detection_data))

            ############ tracking ##############
            tracker.predict()
            tracker.update(detections, t_, frame_name, opt.shot)

            ############ record the results ##############
            final_visuals_dic.setdefault(frame_name, {"time": t_, "shot": opt.shot})
            if opt.render:
                final_visuals_dic[frame_name]["frame"] = image_frame
            for key_ in visual_store_:
                final_visuals_dic[frame_name][key_] = []

            for tracks_ in tracker.tracks:
                if frame_name not in tracked_frames:
                    tracked_frames.append(frame_name)
                if not (tracks_.is_confirmed()):
                    continue

                track_id = tracks_.track_id
                track_data_hist = tracks_.track_data["history"][-1]
                track_data_pred = tracks_.track_data["prediction"]

                final_visuals_dic[frame_name]["tid"].append(track_id)
                final_visuals_dic[frame_name]["bbox"].append(track_data_hist["bbox"])
                final_visuals_dic[frame_name]["tracked_time"].append(
                    tracks_.time_since_update
                )

                for hkey_ in history_keys:
                    final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
                for pkey_ in prediction_keys:
                    final_visuals_dic[frame_name][pkey_].append(
                        track_data_pred[pkey_.split("_")[1]][-1]
                    )

                if tracks_.time_since_update == 0:
                    final_visuals_dic[frame_name]["tracked_ids"].append(track_id)
                    final_visuals_dic[frame_name]["tracked_bbox"].append(
                        track_data_hist["bbox"]
                    )

                    if tracks_.hits == opt.n_init:
                        for pt in range(opt.n_init - 1):
                            track_data_hist_ = tracks_.track_data["history"][-2 - pt]
                            track_data_pred_ = tracks_.track_data["prediction"]
                            frame_name_ = tracked_frames[-2 - pt]
                            final_visuals_dic[frame_name_]["tid"].append(track_id)
                            final_visuals_dic[frame_name_]["bbox"].append(
                                track_data_hist_["bbox"]
                            )
                            final_visuals_dic[frame_name_]["tracked_ids"].append(
                                track_id
                            )
                            final_visuals_dic[frame_name_]["tracked_bbox"].append(
                                track_data_hist_["bbox"]
                            )
                            final_visuals_dic[frame_name_]["tracked_time"].append(0)

                            for hkey_ in history_keys:
                                final_visuals_dic[frame_name_][hkey_].append(
                                    track_data_hist_[hkey_]
                                )
                            for pkey_ in prediction_keys:
                                final_visuals_dic[frame_name_][pkey_].append(
                                    track_data_pred_[pkey_.split("_")[1]][-1]
                                )

            ############ save the video ##############
            if opt.render and t_ >= opt.n_init:
                video_path = f"{vis_dir}/{opt.video_seq}_{opt.detection_type}.mp4"
                d_ = opt.n_init + 1 if (t_ + 1 == len(list_of_frames)) else 1
                for t__ in range(t_, t_ + d_):
                    frame_key = list_of_frames[t__ - opt.n_init]
                    rendered_, f_size = render_frame_main_online(
                        opt,
                        phalp_tracker,
                        frame_key,
                        final_visuals_dic[frame_key],
                        opt.track_dataset,
                        track_id=-100,
                    )
                    if t__ - opt.n_init in list_of_shots:
                        cv2.rectangle(
                            rendered_, (0, 0), (f_size[0], f_size[1]), (0, 0, 255), 4
                        )
                    if t__ - opt.n_init == 0:
                        writer = imageio.get_writer(video_path, fps=30)

                    writer.append_data(rendered_[..., ::-1])
                    del final_visuals_dic[frame_key]["frame"]
                    for tkey_ in tmp_keys_:
                        del final_visuals_dic[frame_key][tkey_]

        joblib.dump(final_visuals_dic, res_file)
        if opt.use_gt:
            gt_file = f"{res_dir}/{opt.video_seq}_{opt.start_frame}_distance.pkl"
            joblib.dump(tracker.tracked_cost, gt_file)
        if opt.render:
            writer.close()

    except Exception as e:
        print(e)
        print(traceback.format_exc())


class options:
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="PHALP_pixel Tracker")
        self.parser.add_argument("--base_path", type=str, required=True)
        self.parser.add_argument("--storage_folder", type=str, default="outputs")
        self.parser.add_argument("--batch_id", type=int, default="-1")
        self.parser.add_argument("--track_dataset", type=str, default="posetrack")
        self.parser.add_argument("--video_seq", type=str, required=True)
        self.parser.add_argument("--sample", type=str, default="")
        self.parser.add_argument("--predict", type=str, default="TPL")
        self.parser.add_argument("--distance_type", type=str, default="EQ_010")
        self.parser.add_argument(
            "--use_gt", type=str2bool, nargs="?", const=True, default=False
        )
        self.parser.add_argument(
            "--overwrite", type=str2bool, nargs="?", const=True, default=False
        )

        self.parser.add_argument("--alpha", type=float, default=0.1)
        self.parser.add_argument("--low_th_c", type=float, default=0.8)
        self.parser.add_argument("--hungarian_th", type=float, default=100.0)
        self.parser.add_argument("--track_history", type=int, default=7)
        self.parser.add_argument("--max_age_track", type=int, default=50)
        self.parser.add_argument("--n_init", type=int, default=5)
        self.parser.add_argument("--max_ids", type=int, default=50)
        self.parser.add_argument(
            "--verbose", type=str2bool, nargs="?", const=True, default=False
        )
        self.parser.add_argument(
            "--detect_shots", type=str2bool, nargs="?", const=True, default=True
        )

        self.parser.add_argument(
            "--store_mask", type=str2bool, nargs="?", const=True, default=True
        )

        self.parser.add_argument(
            "--render", type=str2bool, nargs="?", const=True, default=False
        )
        self.parser.add_argument("--render_type", type=str, default="HUMAN_FULL_FAST")
        self.parser.add_argument("--render_up_scale", type=int, default=2)
        self.parser.add_argument("--res", type=int, default=256)
        self.parser.add_argument("--downsample", type=int, default=1)

        self.parser.add_argument("--encode_type", type=str, default="4c")
        self.parser.add_argument("--cva_type", type=str, default="least_square")
        self.parser.add_argument("--past_lookback", type=int, default=1)
        self.parser.add_argument("--mask_type", type=str, default="feat")
        self.parser.add_argument("--detection_type", type=str, default="mask2")
        self.parser.add_argument("--start_frame", type=int, default=-1)
        self.parser.add_argument("--end_frame", type=int, default=-1)
        self.parser.add_argument(
            "--store_extra_info", type=str2bool, nargs="?", const=True, default=False
        )

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


if __name__ == "__main__":

    opt = options().parse()

    phalp_tracker = PHALP_tracker(opt)
    phalp_tracker.cuda()
    phalp_tracker.eval()

    test_tracker(opt, phalp_tracker)
