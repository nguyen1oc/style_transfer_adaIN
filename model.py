import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


## Support function
def calc_mean_std(_feat):
    """
    input: feature params => [batch_size, c, h, w]
    output: feature params => [batch_size, c, 1, 1]
    """
    mean = _feat.mean([2,3], keepdim=True)
    std = _feat.std([2,3], keepdim=True)
    return mean, std
    
class VGG_encoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = vgg19(pretrained=True).features

        self.enc_1 = vgg[:2]     # relu1_1
        self.enc_2 = vgg[2:7]    # relu2_1
        self.enc_3 = vgg[7:12]   # relu3_1
        self.enc_4 = vgg[12:21]  # relu4_1
        for p in vgg.parameters():
            p.requires_grad = False

    def forward(self, x, last_only=False):
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)

        if last_only:
            return h4
        else:
            return (h1, h2, h3, h4)

def AdaIN_layer(content_feat, style_feat, eps = 1e-5):
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    return s_std * (content_feat - c_mean)/(c_std+eps) + s_mean

class RC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated = True):
        super().__init__()
        self.pad=nn.ReflectionPad2d((pad_size,pad_size,pad_size,pad_size))
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size)
        self.activated=activated

    def forward(self,x):
        h=self.pad(x)
        h=self.conv(h)
        if self.activated:
            return F.relu(h)
        return h
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1=RC(512,256,3,1)
        self.rc2=RC(256,256,3,1)
        self.rc3=RC(256,256,3,1)
        self.rc4=RC(256,256,3,1)
        self.rc5=RC(256,128,3,1)
        self.rc6=RC(128,128,3,1)
        self.rc7=RC(128,64,3,1)
        self.rc8=RC(64,64,3,1)
        self.rc9=RC(64,3,3,1, False)

    def forward(self,features):
        h=self.rc1(features) 
        h=F.interpolate(h,scale_factor=2)
        h=self.rc2(h)
        h=self.rc3(h)
        h=self.rc4(h)
        h=self.rc5(h)
        h=F.interpolate(h, scale_factor=2)
        h=self.rc6(h)
        h=self.rc7(h)
        h=F.interpolate(h, scale_factor=2)
        h=self.rc8(h)
        h=self.rc9(h)
        return h
    
class Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.encoder = VGG_encoder()
        self.decoder = Decoder()
        self.to(device)

    def content_loss(self,out_feat, target_feat):
        return F.mse_loss(out_feat, target_feat)

    def style_loss(self, out_feat, style_feat):
        loss = 0
        for f_out, f_style in zip(out_feat, style_feat):
            out_mean, out_std = calc_mean_std(f_out)
            style_mean, style_std = calc_mean_std(f_style)

            loss += F.mse_loss(out_mean, style_mean)
            loss += F.mse_loss(out_std, style_std)

        return loss

    def generate(self, content_img, style_img, alpha = 1.0):
        #Encoder
        content_feat = self.encoder(content_img, True)
        style_feat = self.encoder(style_img, True)
    
        #AdaIN
        t = AdaIN_layer(content_feat, style_feat)
        t = alpha * t + (1-alpha)*content_feat
    
        #Decoder
        T = self.decoder(t)
        return T
        
    def forward(self, content_img, style_img, alpha = 1, lamb = 10):
        """
            content_img, style_img: [B,3,H,W] normalized
            alpha: blend factor
        """
        #Encoder
        content_feat = self.encoder(content_img, True)
        style_feat = self.encoder(style_img, True)
    
        #AdaIN
        t = AdaIN_layer(content_feat, style_feat)
        t = alpha * t + (1-alpha)*content_feat
    
        #Decoder
        T = self.decoder(t)

        #Encoder
        out_last = self.encoder(T, last_only=True)
        out_m_feat = self.encoder(T, last_only=False)
        style_m_feat = self.encoder(style_img, last_only=False)
        
        loss_c = self.content_loss(out_last, t)
        loss_s = self.style_loss(out_m_feat, style_m_feat)

        total_loss = loss_c + lamb * loss_s
        return total_loss