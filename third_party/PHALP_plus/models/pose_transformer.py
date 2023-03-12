import torch.nn as nn
from .heads.smpl_head_prediction import SMPLHeadPrediction
from .transformers import RelationTransformerModel

from yacs.config import CfgNode as CN

class Pose_transformer(nn.Module):
    
    def __init__(self, opt):
        super(Pose_transformer, self).__init__()
        
        config = "utils/config.yaml"
        with open(config, 'r') as f:
            cfg = CN.load_cfg(f); cfg.freeze()

        self.cfg                   = cfg
        self.relational            = RelationTransformerModel(cfg.MODEL.TRANSFORMER)  
        self.smpl_head_prediction  = SMPLHeadPrediction(cfg)      
    
