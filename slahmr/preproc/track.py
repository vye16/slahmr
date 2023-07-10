import warnings
from dataclasses import dataclass
from typing import Optional
import numpy as np

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger

from vitpose_model import ViTPoseModel

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)

class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:,:3,:,:],
            'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)

        # Overriding the SMPL params with the HMR2 params
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out

class PHALP_Prime_HMR2(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        # initialize vitpose model
        self.ViTPose = ViTPoseModel('cuda')
        self.HMAR = HMR2Predictor(self.cfg)

    def run_additional_models(self, image_frame, pred_bbox, pred_masks, pred_scores, pred_classes, frame_name, t_, measurments, gt_tids, gt_annots):
        vitposes_out = self.ViTPose.predict_pose(
            image_frame[:, :, ::-1],
            [np.concatenate([pred_bbox, pred_scores[:, None]], axis=1)],
        )
        vitposes_list = []
        for vitpose in vitposes_out:
            vitpose_2d = np.zeros([25, 3])
            vitpose_2d[
                [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
            ] = vitpose["keypoints"]
            vitposes_list.append(vitpose_2d)
        return vitposes_list

@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    pass

cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:

    """Main function for running the PHALP tracker."""

    phalp_tracker = PHALP_Prime_HMR2(cfg)

    phalp_tracker.track()

if __name__ == "__main__":
    main()
