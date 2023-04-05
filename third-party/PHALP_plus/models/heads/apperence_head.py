import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
 
class TextureHead(nn.Module):

    def __init__(self, uv_sampler, opts, img_H=64, img_W=128, n_upconv=6, nc_init=32, predict_flow=False, symmetric=False, num_sym_faces=624):
        super(TextureHead, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init

        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)

        nc_final=2
        
        self.decoder = decoder(n_upconv, nc_init, nc_final, self.feat_H, self.feat_W)

    def forward(self, skips):
        
        flow = self.decoder.forward(skips)
        flow = torch.nn.functional.tanh(flow)
        
        return flow


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
                
                
                


class decoder(nn.Module):
    def __init__(self, n_upconv, nc_init, nc_final, H, W):
        super().__init__()
        
        self.n_upconv = n_upconv
        self.nc_init  = nc_init
        self.nc_final = nc_final
        self.feat_H = H
        self.feat_W = W
        nf = 8
        
        self.G_middle_0 = ResnetBlock(2048, 8 * nf)
        self.G_middle_1 = ResnetBlock(8 * nf, 8 * nf)

        self.G_middle_2 = ResnetBlock(1024, 8 * nf)
        self.G_middle_2_1 = ResnetBlock(2*8 * nf, 8 * nf)
        
        self.G_middle_3 = ResnetBlock(512, 8 * nf)
        self.G_middle_3_1 = ResnetBlock(2*8 * nf, 8 * nf)
        
        self.G_middle_4 = ResnetBlock(256, 4 * nf)
        self.G_middle_4_1 = ResnetBlock(2*4 * nf, 4 * nf)
        
        
        self.up_0 = ResnetBlock(8 * nf, 4 * nf)
        self.up_1 = ResnetBlock(4 * nf, 4 * nf)
        self.up_2 = ResnetBlock(4 * nf, 2 * nf)
        self.up_3 = ResnetBlock(2 * nf, 1 * nf)

        self.conv_img = nn.Conv2d(nf, self.nc_final, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        
        
        
    def forward(self, skips):
        x1,x2,x3,x4 = skips
        x = x4
        x = self.G_middle_0(x)

        if self.n_upconv >= 6:
            x = self.up(x)

        x = torch.cat([x, self.G_middle_2(x3)], 1)
        x = self.G_middle_2_1(x)
        x = self.G_middle_1(x)

        x = self.up(x)
        x = torch.cat([x, self.G_middle_3(x2)], 1)
        x = self.G_middle_3_1(x)
        x = self.up_0(x)
        
        x = self.up(x)
        x = torch.cat([x, self.G_middle_4(x1)], 1)
        x = self.G_middle_4_1(x)
        x = self.up_1(x)
        
        x = self.up(x)
        x = self.up_2(x)
        x = self.up(x)
        x = self.up_3(x)

        if self.n_upconv >= 7:
            x = self.up(x)
            x = self.up_4(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        
        return x




class ResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)
            
        # define normalization layers
        self.norm_0 = BatchNorm2d(fin)
        self.norm_1 = BatchNorm2d(fmiddle)
        if self.learned_shortcut:
            self.norm_s = BatchNorm2d(fin)

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv_0(self.actvn(self.norm_0(x)))
        dx = self.conv_1(self.actvn(self.norm_1(dx)))

        out = x_s + dx

        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
    
    