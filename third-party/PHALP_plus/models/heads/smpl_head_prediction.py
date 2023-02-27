import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import BatchNorm2d

class SMPLHeadPrediction(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, cfg):
        super(SMPLHeadPrediction, self).__init__()
        self.cfg = cfg
        in_channels = cfg.MODEL.SMPL_HEAD.IN_CHANNELS

        self.up1_1 = nn.Linear(in_channels, in_channels)
        self.dropup1 = nn.Dropout()
        self.up2_1 = nn.Linear(in_channels, in_channels)
        self.up1_2 = nn.Linear(in_channels, in_channels)
        self.dropup2 = nn.Dropout()
        self.up2_2 = nn.Linear(in_channels, in_channels)
        self.up1_3 = nn.Linear(in_channels, in_channels)
        self.dropup3 = nn.Dropout()
        self.up2_3 = nn.Linear(in_channels, in_channels)
        self.up1_4 = nn.Linear(in_channels, in_channels)
        self.dropup4 = nn.Dropout()
        self.up2_4 = nn.Linear(in_channels, in_channels)
        self.up1_5 = nn.Linear(in_channels, in_channels)
        self.dropup5 = nn.Dropout()
        self.up2_5 = nn.Linear(in_channels, in_channels)
        self.up1_6 = nn.Linear(in_channels, in_channels)
        self.dropup6 = nn.Dropout()
        self.up2_6 = nn.Linear(in_channels, in_channels)
        self.up1_7 = nn.Linear(in_channels, in_channels)
        self.dropup7 = nn.Dropout()
        self.up2_7 = nn.Linear(in_channels, in_channels)
        self.up1_8 = nn.Linear(in_channels, in_channels)
        self.dropup8 = nn.Dropout()
        self.up2_8 = nn.Linear(in_channels, in_channels)
        self.up1_9 = nn.Linear(in_channels, in_channels)
        self.dropup9 = nn.Dropout()
        self.up2_9 = nn.Linear(in_channels, in_channels)
        self.up1_10 = nn.Linear(in_channels, in_channels)
        self.dropup10 = nn.Dropout()
        self.up2_10 = nn.Linear(in_channels, in_channels)
        self.up1_11 = nn.Linear(in_channels, in_channels)
        self.dropup11 = nn.Dropout()
        self.up2_11 = nn.Linear(in_channels, in_channels)
        self.up1_12 = nn.Linear(in_channels, in_channels)
        self.dropup12 = nn.Dropout()
        self.up2_12 = nn.Linear(in_channels, in_channels)
        nn.init.xavier_uniform_(self.up2_1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_2.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_3.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_4.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_5.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_6.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_7.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_8.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_9.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_10.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_11.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up2_12.weight, gain=0.01)

    def forward(self, x, xt):
        
        xf0 = x
        xhmmr = x + xt
        xf = F.relu(self.up1_1(xhmmr))
        xf = self.dropup1(xf)
        xf1 = xf0 + self.up2_1(xf)

        xf = F.relu(self.up1_2(xhmmr))
        xf = self.dropup2(xf)
        xf2 = xf1 + self.up2_2(xf)

        xf = F.relu(self.up1_3(xhmmr))
        xf = self.dropup3(xf)
        xf3 = xf2 + self.up2_3(xf)

        xf = F.relu(self.up1_4(xhmmr))
        xf = self.dropup4(xf)
        xf4 = xf3 + self.up2_4(xf)

        xf = F.relu(self.up1_5(xhmmr))
        xf = self.dropup5(xf)
        xf5 = xf4 + self.up2_5(xf)

        xf = F.relu(self.up1_6(xhmmr))
        xf = self.dropup6(xf)
        xf6 = xf5 + self.up2_6(xf)

        xf = F.relu(self.up1_7(xhmmr))
        xf = self.dropup7(xf)
        xf7 = xf6 + self.up2_7(xf)

        xf = F.relu(self.up1_8(xhmmr))
        xf = self.dropup8(xf)
        xf8 = xf7 + self.up2_8(xf)

        xf = F.relu(self.up1_9(xhmmr))
        xf = self.dropup9(xf)
        xf9 = xf8 + self.up2_9(xf)

        xf = F.relu(self.up1_10(xhmmr))
        xf = self.dropup10(xf)
        xf10 = xf9 + self.up1_10(xf)

        xf = F.relu(self.up1_11(xhmmr))
        xf = self.dropup11(xf)
        xf11 = xf10 + self.up2_11(xf)

        xf = F.relu(self.up1_12(xhmmr))
        xf = self.dropup12(xf)
        xf12 = xf11 + self.up2_12(xf)

        xf = torch.cat([xf1[:,None], xf2[:,None], xf3[:,None], xf4[:,None], xf5[:,None], xf6[:,None], xf7[:,None], xf8[:,None], xf9[:,None], xf10[:,None], xf11[:,None], xf12[:,None]], dim=1)

        return xf
