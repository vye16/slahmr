import torch
import torch.nn as nn
import torch.nn.functional as F
    
class EncodingHead(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, opt, img_H=64, img_W=128):
        super(EncodingHead, self).__init__()
        
        self.opt = opt
        
        #Encoder
        if(self.opt.encode_type=="3c"): self.conv1   = nn.Conv2d(3, 16, 3, padding=1)  
        if(self.opt.encode_type=="4c"): self.conv1x  = nn.Conv2d(4, 16, 3, padding=1)  
        self.conv2   = nn.Conv2d(16, 64, 3, padding=1)
        self.conv3   = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4   = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5   = nn.Conv2d(256, 256, 3, padding=1, stride=1)
        self.conv5_1 = nn.Conv2d(256, 16, 3, padding=1, stride=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        
        
        #Decoder
        self.t_conv5_1  = nn.Conv2d(16, 256, 3, stride=1,  padding=1)
        self.t_bn_5_1   = nn.BatchNorm2d(256)
         
        self.t_conv5    = nn.Conv2d(256, 256, 3, stride=1,  padding=1)
        self.t_bn_5     = nn.BatchNorm2d(256)
 
        self.t_conv4    = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.t_bn_4     = nn.BatchNorm2d(128)
         
        self.t_conv3    = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.t_bn_3     = nn.BatchNorm2d(64)
         
        self.t_conv2    = nn.Conv2d(64, 16, 3, stride=1, padding=1)
        self.t_bn_2     = nn.BatchNorm2d(16)
         
        if(self.opt.encode_type=="3c"):
            self.t_conv1    = nn.Conv2d(16, 3, 3, stride=1, padding=1)
            self.t_bn_1     = nn.BatchNorm2d(3)
            self.t_conv1_1  = nn.Conv2d(3, 3, 3, stride=1, padding=1)
        
        if(self.opt.encode_type=="4c"):
            self.t_conv1x   = nn.Conv2d(16, 4, 3, stride=1, padding=1)
            self.t_bn_1x    = nn.BatchNorm2d(4)
            self.t_conv1_1x = nn.Conv2d(4, 4, 3, stride=1, padding=1)


    def forward(self, x, en=True):
        
        if(en):
            if(self.opt.encode_type=="3c"): x = F.relu(self.conv1(x))
            if(self.opt.encode_type=="4c"): x = F.relu(self.conv1x(x))
            x = self.pool(x)

            x = F.relu(self.conv2(x))
            x = self.pool(x)

            x = F.relu(self.conv3(x))
            x = self.pool(x)

            x = F.relu(self.conv4(x))
            x = self.pool(x)

            x = F.relu(self.conv5(x))  
            x = F.relu(self.conv5_1(x)) 
            return x
        
        else:
            x = F.relu(self.t_bn_5_1(self.t_conv5_1(x)))
            x = F.relu(self.t_bn_5(self.t_conv5(x)))

            x = F.relu(self.t_bn_4(self.t_conv4(x)))
            x = self.up(x)

            x = F.relu(self.t_bn_3(self.t_conv3(x)))
            x = self.up(x)

            x = F.relu(self.t_bn_2(self.t_conv2(x)))
            x = self.up(x)
            
            if(self.opt.encode_type=="3c"):
                x = F.relu(self.t_bn_1(self.t_conv1(x)))
                x = self.up(x)
                x = F.tanh(self.t_conv1_1(x))
            
            if(self.opt.encode_type=="4c"):
                x = F.relu(self.t_bn_1x(self.t_conv1x(x)))
                x = self.up(x)
                x = F.tanh(self.t_conv1_1x(x))
            
            return x
        
        
