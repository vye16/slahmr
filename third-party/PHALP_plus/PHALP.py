import torch
import torch.nn as nn

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

import os
import cv2
import gdown, dill
import numpy as np
from sklearn.linear_model import Ridge

from models.hmar import HMAR
from models.utils import *

from detectron2.config import LazyConfig
from utils.utils_detectron2 import DefaultPredictor_Lazy
from utils.utils_dataset import process_image, process_mask
from utils.utils import get_prediction_interval
from scenedetect import AdaptiveDetector
from utils.utils_scenedetect import detect

class PHALP_tracker(nn.Module):
    def __init__(self, opt):
        super(PHALP_tracker, self).__init__()

        # download wights and configs from Google Drive
        self.cached_download_from_drive()

        self.opt = opt
        self.HMAR = HMAR("utils/config.yaml", self.opt)
        self.device = torch.device("cuda")
        checkpoint_file = torch.load("_DATA/hmar_v2_weights.pth")
        state_dict_filt = {}
        for k, v in checkpoint_file["model"].items():
            if (
                "encoding_head" in k
                or "texture_head" in k
                or "backbone" in k
                or "smplx_head" in k
            ):
                state_dict_filt.setdefault(k[5:].replace("smplx", "smpl"), v)
        self.HMAR.load_state_dict(state_dict_filt, strict=False)
        self.HMAR.to(self.device)
        self.HMAR.eval()

        #self.detectron2_cfg = model_zoo.get_config(
        #    "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        #)
        #self.detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        #self.detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4

        self.detectron2_cfg = LazyConfig.load(
            'utils/cascade_mask_rcnn_vitdet_h_75ep.py'
        )
        self.detectron2_cfg.train.init_checkpoint = 'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl'
        for i in range(3):
            self.detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
        
        self.detector = DefaultPredictor_Lazy(self.detectron2_cfg)
        self.detectron2_cfg = get_cfg()
        self.detectron2_cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            )
        )

    def forward_for_tracking(self, vectors, attibute="A", time=1):

        if attibute == "P":

            vectors_pose = vectors[0]
            vectors_data = vectors[1]
            vectors_time = vectors[2]

            en_pose = torch.from_numpy(vectors_pose)
            en_data = torch.from_numpy(vectors_data)
            en_time = torch.from_numpy(vectors_time)

            if len(en_pose.shape) != 3:
                en_pose = en_pose.unsqueeze(0)  # (BS, 7, 4096)
                en_time = en_time.unsqueeze(0)  # (BS, 7)
                en_data = en_data.unsqueeze(0)  # (BS, 7, 6)

            BS = en_pose.size(0)
            history = en_pose.size(1)
            attn = torch.ones(BS, history, history)

            xf_trans = self.HMAR.pose_transformer.relational(
                en_pose[:, :, 2048:].float().cuda(),
                en_data.float().cuda(),
                attn.float().cuda(),
            )  # bs, 13, 2048
            xf_trans = xf_trans.view(-1, 2048)
            movie_strip_t = self.HMAR.pose_transformer.smpl_head_prediction(
                en_pose[:, :, 2048:].float().view(-1, 2048).cuda(), xf_trans
            )  # bs*13, 2048 -> bs*13, 12, 2048
            movie_strip_t = movie_strip_t.view(BS, history, 12, 2048)
            xf_trans = xf_trans.view(BS, history, 2048)

            time[time > 11] = 11
            pose_pred = []
            for i in range(len(time)):
                pose_pred.append(movie_strip_t[i, -1, time[i], :])
            pose_pred = torch.stack(pose_pred)
            en_pose_x = torch.cat((xf_trans[:, -1, :], pose_pred), 1)

            return en_pose_x.cpu()

        if attibute == "L":
            vectors_loca = vectors[0]
            vectors_time = vectors[1]
            vectors_conf = vectors[2]

            en_loca = torch.from_numpy(vectors_loca)
            en_time = torch.from_numpy(vectors_time)
            en_conf = torch.from_numpy(vectors_conf)
            time = torch.from_numpy(time)

            if len(en_loca.shape) != 3:
                en_loca = en_loca.unsqueeze(0)
                en_time = en_time.unsqueeze(0)
            else:
                en_loca = en_loca.permute(0, 1, 2)

            BS = en_loca.size(0)
            t_ = en_loca.size(1)

            en_loca_xy = en_loca[:, :, :90]
            en_loca_xy = en_loca_xy.view(BS, t_, 45, 2)
            en_loca_n = en_loca[:, :, 90:]
            en_loca_n = en_loca_n.view(BS, t_, 3, 3)

            if self.opt.cva_type == "least_square":
                new_en_loca_n = []
                for bs in range(BS):
                    x0_ = np.array(en_loca_xy[bs, :, 44, 0])
                    y0_ = np.array(en_loca_xy[bs, :, 44, 1])

                    x_ = np.array(en_loca_n[bs, :, 0, 0])
                    y_ = np.array(en_loca_n[bs, :, 0, 1])
                    n_ = np.log(np.array(en_loca_n[bs, :, 0, 2]))
                    t_ = np.array(en_time[bs, :])
                    n = len(t_)

                    loc_ = torch.diff(en_time[bs, :], dim=0) != 0
                    loc_ = loc_.shape[0] - torch.sum(loc_) + 1

                    M = t_[:, np.newaxis] ** [0, 1]
                    time_ = 48 if time[bs] > 48 else time[bs]

                    clf = Ridge(alpha=5.0)
                    clf.fit(M, n_)
                    n_p = clf.predict(np.array([1, time_ + 1 + t_[-1]]).reshape(1, -1))
                    n_p = n_p[0]
                    n_hat = clf.predict(
                        np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1))))
                    )
                    n_pi = get_prediction_interval(n_, n_hat, t_, time_ + 1 + t_[-1])

                    clf = Ridge(alpha=1.2)
                    clf.fit(M, x0_)
                    x_p = clf.predict(np.array([1, time_ + 1 + t_[-1]]).reshape(1, -1))
                    x_p = x_p[0]
                    x_p_ = (x_p - 0.5) * np.exp(n_p) / 5000.0 * 256.0
                    x_hat = clf.predict(
                        np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1))))
                    )
                    x_pi = get_prediction_interval(x0_, x_hat, t_, time_ + 1 + t_[-1])

                    clf = Ridge(alpha=2.0)
                    clf.fit(M, y0_)
                    y_p = clf.predict(np.array([1, time_ + 1 + t_[-1]]).reshape(1, -1))
                    y_p = y_p[0]
                    y_p_ = (y_p - 0.5) * np.exp(n_p) / 5000.0 * 256.0
                    y_hat = clf.predict(
                        np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1))))
                    )
                    y_pi = get_prediction_interval(y0_, y_hat, t_, time_ + 1 + t_[-1])

                    new_en_loca_n.append(
                        [
                            x_p_,
                            y_p_,
                            np.exp(n_p),
                            x_pi / loc_,
                            y_pi / loc_,
                            np.exp(n_pi) / loc_,
                            1,
                            1,
                            0,
                        ]
                    )
                    en_loca_xy[bs, -1, 44, 0] = x_p
                    en_loca_xy[bs, -1, 44, 1] = y_p

                new_en_loca_n = torch.from_numpy(np.array(new_en_loca_n))
                xt = torch.cat(
                    (
                        en_loca_xy[:, -1, :, :].view(BS, 90),
                        (new_en_loca_n.float()).view(BS, 9),
                    ),
                    1,
                )

        return xt

    def get_uv_distance(self, t_uv, d_uv):
        t_uv = torch.from_numpy(t_uv).cuda().float()
        d_uv = torch.from_numpy(d_uv).cuda().float()
        d_mask = d_uv[3:, :, :] > 0.5
        t_mask = t_uv[3:, :, :] > 0.5

        mask = torch.logical_and(d_mask, t_mask)
        mask = mask.repeat(4, 1, 1)
        mask_ = torch.logical_not(mask)

        t_uv[mask_] = 0.0
        d_uv[mask_] = 0.0

        with torch.no_grad():
            t_emb = self.HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
            d_emb = self.HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
        t_emb = t_emb.view(-1) / 10**3
        d_emb = d_emb.view(-1) / 10**3
        return (
            t_emb.cpu().numpy(),
            d_emb.cpu().numpy(),
            torch.sum(mask).cpu().numpy() / 4 / 256 / 256 / 2,
        )

    def get_human_apl(
        self, image, mask, bbox, score, frame_name, mask_name, t_, measurments, gt=1
    ):
        self.HMAR.reset_nmr(self.opt.res)

        img_height, img_width, new_image_size, left, top = measurments
        mask_a = mask.astype(int) * 255

        if len(mask_a.shape) == 2:
            mask_a = np.expand_dims(mask_a, 2)
            mask_a = np.repeat(mask_a, 3, 2)

        center_ = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2])
        scale_ = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])
        mask_tmp = process_mask(mask_a.astype(np.uint8), center_, 1.0 * np.max(scale_))
        image_tmp = process_image(image, center_, 1.0 * np.max(scale_))
        masked_image = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)
        ratio = 1.0 / int(new_image_size) * self.opt.res

        with torch.no_grad():
            hmar_out = self.HMAR(masked_image.unsqueeze(0).cuda())

            uv_image = hmar_out["uv_image"][:, :3, :, :] / 5.0
            uv_mask = hmar_out["uv_image"][:, 3:, :, :]
            zeros_ = uv_mask == 0
            ones_ = torch.logical_not(zeros_)
            zeros_ = zeros_.repeat(1, 3, 1, 1)
            ones_ = ones_.repeat(1, 3, 1, 1)
            uv_image[zeros_] = 0.0
            uv_mask[zeros_[:, :1, :, :]] = -1.0
            uv_mask[ones_[:, :1, :, :]] = 1.0
            uv_vector = torch.cat((uv_image, uv_mask), 1)
            pose_embedding = hmar_out["pose_emb"]
            appe_embedding = self.HMAR.autoencoder_hmar(uv_vector, en=True)
            appe_embedding = appe_embedding.view(1, -1)
            _, _, pred_joints_2d, pred_joints, pred_cam = self.HMAR.render_3d(
                torch.cat((pose_embedding, pose_embedding), 1),
                np.array([[1.0, 0, 0]]),
                center=(center_ + [left, top]) * ratio,
                img_size=self.opt.res,
                scale=np.reshape(np.array([max(scale_)]), (1, 1)) * ratio,
                texture=uv_vector[:, :3, :, :] * 5.0,
                render=False,
            )
            pred_smpl_params, pred_cam_x, _ = self.HMAR.smpl_head(
                pose_embedding.float()
            )
            pred_smpl_params = {k: v.cpu().numpy() for k, v in pred_smpl_params.items()}
            pred_joints_2d_ = (
                pred_joints_2d.reshape(
                    -1,
                )
                / self.opt.res
            )
            pred_cam_ = pred_cam.view(
                -1,
            )
            pred_joints_2d_.contiguous()
            pred_cam_.contiguous()
            loca_embedding = torch.cat(
                (pred_joints_2d_, pred_cam_, pred_cam_, pred_cam_), 0
            )

        full_embedding = torch.cat(
            (
                appe_embedding[0].cpu(),
                pose_embedding[0].cpu(),
                pose_embedding[0].cpu(),
                loca_embedding.cpu(),
            ),
            0,
        )

        detection_data = {
            "bbox": np.array(
                [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
            ),
            "conf": score,
            "appe": appe_embedding[0].cpu().numpy(),
            "pose": torch.cat((pose_embedding[0].cpu(), pose_embedding[0].cpu()), 0)
            .cpu()
            .numpy(),
            "loca": loca_embedding.cpu().numpy(),
            "embedding": full_embedding,
            "uv": uv_vector[0].cpu().numpy(),
            "center": center_,
            "scale": scale_,
            "smpl": pred_smpl_params,
            "3d_joints": pred_joints[0].cpu().numpy(),
            "2d_joints": pred_joints[0].cpu().numpy(),
            "camera": pred_cam_x.cpu().numpy(),
            "size": [img_height, img_width],
            "img_path": frame_name[0] + "/" + frame_name[1],
            "img_name": frame_name[1],
            "mask_name": mask_name,
            "ground_truth": gt,
            "time": t_,
        }

        return detection_data

    def get_detections(self, image, frame_name, t_):
        image_to_write = image.copy()
        mask_names = []

        if "mask" in self.opt.detection_type:
            outputs = self.detector(image)
            instances = outputs["instances"]
            instances = instances[instances.pred_classes == 0]
            instances = instances[instances.scores > self.opt.low_th_c]

            if self.opt.render:
                visualizer = Visualizer(
                    image_to_write[:, :, ::-1],
                    MetadataCatalog.get(self.detectron2_cfg.DATASETS.TRAIN[0]),
                    scale=1.2,
                )
                if self.opt.store_mask:
                    cv2.imwrite(
                        os.path.join(
                            self.opt.storage_folder,
                            "_TMP",
                            f"{self.opt.video_seq}_{frame_name}",
                        ),
                        visualizer.draw_instance_predictions(
                            instances.to("cpu")
                        ).get_image()[:, :, ::-1],
                    )

            for i in range(instances.pred_classes.shape[0]):
                name = frame_name.split(".")[0]
                mask_name_ = os.path.join(
                    self.opt.storage_folder,
                    "_TMP",
                    f"{self.opt.video_seq}_{name}_{i}.png",
                )
                mask_names.append(mask_name_)
                if self.opt.store_mask:
                    mask_bw = instances.pred_masks[i].cpu().numpy()
                    cv2.imwrite(mask_name_, mask_bw.astype(int) * 255)

            pred_bbox = instances.pred_boxes.tensor.cpu().numpy()
            pred_masks = instances.pred_masks.cpu().numpy()
            pred_scores = instances.scores.cpu().numpy()

        ground_truth = [1 for i in list(range(len(pred_scores)))]

        return pred_bbox, pred_masks, pred_scores, mask_names, ground_truth

    def get_list_of_shots(self, main_path_to_frames, list_of_frames):
        list_of_shots = []
        if self.opt.detect_shots:
            video_tmp_name = os.path.join(
                self.opt.storage_folder, "_TMP", f"{self.opt.video_seq}.mp4"
            )
            for ft_, fname_ in enumerate(list_of_frames):
                im_ = cv2.imread(main_path_to_frames + "/" + fname_)
                if ft_ == 0:
                    video_file = cv2.VideoWriter(
                        video_tmp_name,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        24,
                        frameSize=(im_.shape[1], im_.shape[0]),
                    )
                video_file.write(im_)
            video_file.release()
            try:
                scene_list = detect(video_tmp_name, AdaptiveDetector())
            except:
                pass
            os.system("rm " + video_tmp_name)
            for scene in scene_list:
                print(scene)
                list_of_shots.append(scene[0].get_frames())
                list_of_shots.append(scene[1].get_frames())
            list_of_shots = np.unique(list_of_shots)
            list_of_shots = list_of_shots[1:-1]
        return list_of_shots

    def cached_download_from_drive(self):
        """Download a file from Google Drive if it doesn't exist yet.
        :param url: the URL of the file to download
        :param path: the path to save the file to
        """
        os.makedirs("_DATA/", exist_ok=True)

        if not os.path.exists("_DATA/models/smpl/SMPL_NEUTRAL.pkl"):
            # We are downloading the SMPL model here for convenience. Please accept the license
            # agreement on the SMPL website: https://smpl.is.tue.mpg.
            os.system("mkdir -p _DATA/models")
            os.system("mkdir -p _DATA/models/smpl")
            # os.system('wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pickle')
            os.system(
                "wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
            )

            self.convert("basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
            os.system("rm basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
            os.system(
                "mv basicModel_neutral_lbs_10_207_0_v1.0.0_p3.pkl _DATA/models/smpl/SMPL_NEUTRAL.pkl"
            )

        download_files = {
            "posetrack_gt_data.pickle": "https://drive.google.com/file/d/1pmtc3l6W8AXScRnhV_KIYqTrzM0-Qb-D/view?usp=sharing",
            "posetrack-val_videos.npy": "https://drive.google.com/file/d/1ln5M1Lro7mKH-IQJ4kA0Uj0Xks9apQOH/view?usp=sharing",
            "texture.npz": "https://drive.google.com/file/d/1T37ym8d6tDxLpdOaCJyQ9bZ1ejIvJAoH/view?usp=sharing",
            "SMPL_to_J19.pkl": "https://drive.google.com/file/d/1UWsrBc5XH1ZkB_cfIR9aJVGtwE_0NOPP/view?usp=sharing",
            "smpl_mean_params.npz": "https://drive.google.com/file/d/11mMhMmPJqtDNoOQWA_B4neVpOW_3unCE/view?usp=sharing",
            "J_regressor_h36m.npy": "https://drive.google.com/file/d/1I0QZqGJpyP7Hv5BypmxqX60gwjX2nPNn/view?usp=sharing",
            "hmar_v2_weights.pth": "https://drive.google.com/file/d/1_wZcPv8MxPoZyEGA9rI5ayXiB7Fhhj4b/view?usp=sharing",
            "hmmr_v2_weights.pt": "https://drive.google.com/file/d/1hMjFoyVkoHIiYJBndvCoy2fs9T8j-ULU/view?usp=sharing",
        }

        for file_name, url in download_files.items():
            if not os.path.exists("_DATA/" + file_name):
                print("Downloading file: " + file_name)
                output = gdown.cached_download(url, "_DATA/" + file_name, fuzzy=True)

                assert os.path.exists("_DATA/" + file_name), f"{output} does not exist"

    def convert(self, old_pkl):
        # Code adapted from https://github.com/nkolot/ProHMR
        # Convert SMPL pkl file to be compatible with Python 3
        # Script is from https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/
        import pickle

        # Make a name for the new pickle
        new_pkl = os.path.splitext(os.path.basename(old_pkl))[0] + "_p3.pkl"

        # Convert Python 2 "ObjectType" to Python 3 object
        dill._dill._reverse_typemap["ObjectType"] = object

        # Open the pickle using latin1 encoding
        with open(old_pkl, "rb") as f:
            loaded = pickle.load(f, encoding="latin1")

        # Re-save as Python 3 pickle
        with open(new_pkl, "wb") as outfile:
            pickle.dump(loaded, outfile)
