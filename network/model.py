import math
import torch
import torch.nn as nn
from network.resnet import resnet50
from network.pvtv2 import pvt_v2_b2
import torch.nn.functional as F
import os,sys
sys.path.append(os.getcwd())

import torch

import torch.nn as nn


import torchvision

DW_POS=1
DS_POS=2
C_POS=3


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True,groups=1):
        super().__init__()
        
        self.act = act
        if in_c<out_c:
            groups=in_c
        elif in_c>out_c:
            groups=out_c
        else:
            groups=out_c
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride,groups=groups),
            nn.BatchNorm2d(out_c)
        )
        if act==True:
            self.relu = nn.ReLU(inplace=True)
        self.balance=nn.Conv2d(out_c, out_c, 1)

    def forward(self, x):
        
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        x=self.balance(x)
        return x



class FeatureDecoupling(nn.Module):
    def __init__(self, in_channel=1024,out_channels=1024,channel_scale=2,attention=nn.Sigmoid):
        super().__init__()
   
        keys = ['fg', 'bg', 'uc']
        self.cbr_dict = nn.ModuleDict()

        for key in keys:
            self.cbr_dict[key] = nn.Sequential(
                CBR(in_channel, in_channel // channel_scale, kernel_size=3, padding=1),
                CBR(in_channel // channel_scale, out_channels, kernel_size=3, padding=1),
                CBR(out_channels, out_channels, kernel_size=1, padding=0)
            )
        
        self.attention=attention()

    def forward(self, x):
        
        xf=self.attention(x)
        f_fg = self.cbr_dict['fg'](xf)
        
        xb=1-self.attention(x)
        f_bg = self.cbr_dict['bg'](xb)
        
        xu=torch.exp(-torch.abs(f_fg-f_bg))
        f_uc = self.cbr_dict['uc'](xu)
        
        return f_fg, f_bg, f_uc



class FeatureDecouplingPredictionHead(nn.Module):
    def __init__(self, in_channel,up_factor=1,channel_factor=[1,2,4],channel_scale=2):
        super().__init__()
        
        in_channel_factor=channel_factor
        self.up_factor=up_factor
        
        self.branch_fg=nn.ModuleList()
        
        self.branch_bg=nn.ModuleList()

        self.branch_uc=nn.ModuleList()
        
        for i in range(self.up_factor):
            out_channel = in_channel // in_channel_factor[i]
            self.branch_fg.append(
                nn.Sequential(
                    CBR(in_channel, out_channel, kernel_size=3, padding=1),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            )
            self.branch_bg.append(
                nn.Sequential(
                    CBR(in_channel, out_channel, kernel_size=3, padding=1),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            )
            self.branch_uc.append(
                nn.Sequential(
                    CBR(in_channel, out_channel, kernel_size=3, padding=1),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
            )
            in_channel = out_channel
            
        self.mask_fg=nn.Sequential(
                CBR(in_channel, in_channel//channel_scale, kernel_size=3, padding=1),
                nn.Conv2d(in_channel//channel_scale, 1, kernel_size=1),
                nn.Sigmoid()
            )
        self.mask_bg=nn.Sequential(
                CBR(in_channel, in_channel//channel_scale, kernel_size=3, padding=1),
                nn.Conv2d(in_channel//channel_scale, 1, kernel_size=1),
                nn.Sigmoid()
            )
        self.mask_uc=nn.Sequential(
                CBR(in_channel, in_channel//channel_scale, kernel_size=3, padding=1),
                nn.Conv2d(in_channel//channel_scale, 1, kernel_size=1),
                nn.Sigmoid()
            )
        

    def forward(self, f_fg, f_bg, f_uc):
        
        for i in range(self.up_factor):
            f_fg = self.branch_fg[i](f_fg)
            f_bg = self.branch_bg[i](f_bg)
            f_uc = self.branch_uc[i](f_uc)
        
        mask_fg=self.mask_fg(f_fg)
        mask_bg=self.mask_bg(f_bg)
        mask_uc=self.mask_uc(f_uc)
        
        return mask_fg, mask_bg, mask_uc



class ContrastDrivenFeatureAlignment(nn.Module):
    def __init__(self, in_channel, out_channel, channel_scale=2, pos=DS_POS, attn_drop=0., proj_drop=0.,
                 num_filters=3,l_kernel=(3,3),v_kernel=(5,1),h_kernel=(5,1),p_kernel=[2,4]):
        super().__init__()
        
        self.scale = out_channel**-0.5
        
        self.pro=nn.ModuleList([
               nn.Conv2d(out_channel,out_channel,1) for _ in range(4)
            ])
        self.proj_drop = nn.Dropout(proj_drop)
        
        if pos == DS_POS:
            self.position_embeddings = nn.ModuleList([
                CBR(out_channel, out_channel, kernel_size=3, padding=1) for _ in range(3)
            ])
        elif pos == DW_POS:
            self.position_embeddings = nn.ModuleList([
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel) for _ in range(num_filters)
            ])
        else:
            self.position_embeddings = nn.ModuleList([
                nn.Conv2d(out_channel, out_channel, kernel_size=1) for _ in range(num_filters)
            ])
            
        self.similar_filter=nn.Softmax(dim=-1)
        self.dwconv=CBR(out_channel, out_channel, kernel_size=3, padding=1)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.sig=nn.Sigmoid()
        
        self.h_f_fusion_cbr = nn.Sequential(
            CBR(out_channel, out_channel//channel_scale, kernel_size=3, padding=1),
            CBR(out_channel//channel_scale, out_channel, kernel_size=3, padding=1),
        )
        self.d_filter = CBR(out_channel,out_channel,kernel_size=l_kernel[0],padding=l_kernel[0]//2)
        self.v_filter = CBR(out_channel,out_channel,kernel_size=v_kernel,padding=(v_kernel[0]//2,v_kernel[1]//2))
        self.h_filter = CBR(out_channel,out_channel,kernel_size=h_kernel,padding=(h_kernel[0]//2,h_kernel[1]//2))
        self.h_f_cbr = nn.Sequential(
            CBR(out_channel, out_channel//channel_scale, kernel_size=3, padding=1),
            CBR(out_channel//channel_scale, out_channel, kernel_size=3, padding=1),
        )
        
        self.l_f_fusion_cbr = nn.Sequential(
            CBR(out_channel, out_channel//channel_scale, kernel_size=3, padding=1),
            CBR(out_channel//channel_scale, out_channel, kernel_size=3, padding=1),
        )
        self.s_filter = nn.AvgPool2d(kernel_size=p_kernel[0], stride=2, padding=0)
        self.l_filter = nn.AvgPool2d(kernel_size=p_kernel[1],stride=2,padding=1)
        self.l_f_cbr = nn.Sequential(
            CBR(out_channel, out_channel//channel_scale, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            CBR(out_channel//channel_scale, out_channel, kernel_size=3, padding=1),
        )
        
        self.output_cbr = nn.Sequential(
            CBR(out_channel, out_channel, kernel_size=3, padding=1),
            CBR(out_channel, out_channel, kernel_size=3, padding=1)
        )
        
    def forward_pro(self,x,fg,bg,uc):
        
        O=self.pro[0](x)
        F=self.pro[1](fg)
        B=self.pro[2](bg)
        U=self.pro[3](uc)
        
        O=self.proj_drop(O)
        F=self.proj_drop(F)
        B=self.proj_drop(B)
        U=self.proj_drop(U)
        
        return O,F,B,U
    
    def forward_optimize_foreground(self,O,F,B,U):
        
        sig_F = self.sig(F)
        attn_F = sig_F*O
        
        sig_B = 1.0-self.sig(B)
        attn_B = sig_B*O
        
        sig_U = sig_F*sig_B*self.sig(U)
        attn_U = sig_U*O
        
        return attn_F,attn_B,attn_U
    
    def forward_high_frequency(self,attn_F,attn_B,attn_U):
        
        X = attn_F+attn_B+attn_U
        X = self.h_f_fusion_cbr(X)
        X_d = self.d_filter(X)
        X_v = self.v_filter(X)
        X_h = self.h_filter(X)
        X = self.h_f_cbr(X_d+X_v+X_h)*attn_F
        return X
    
    def forward_low_frequency(self,attn_F,attn_B,attn_U):
        
        X = attn_F+attn_B+attn_U
        X = self.l_f_fusion_cbr(X)
        X_d = self.s_filter(X)
        X_v = self.l_filter(X)
        X = self.l_f_cbr(X_d+X_v)*attn_F
        return X
    
    def forward_samiliarity(self,attn_F,attn_B,attn_U):
        
        B,C,H,W=attn_F.shape
        
        q=self.position_embeddings[0](attn_F).view(B,C,H*W)
        k=self.position_embeddings[1](attn_B).view(B,C,H*W)
        v=self.position_embeddings[2](attn_U).view(B,C,H*W)
        
        attn = self.similar_filter(q @ k.transpose(-1, -2) * self.scale)
        attn = attn @ v
        
        attn=self.attn_drop(self.dwconv(attn.view(B,C,H,W)) * attn_F)
        
        return attn
        

    def forward(self, x, fg, bg, uc):
        
        B, C,H, W= x.shape
        
        O,F,B,U = self.forward_pro(x,fg,bg,uc)
        
        attn_F,attn_B,attn_U=self.forward_optimize_foreground(O,F,B,U) 
        
        out_s=self.forward_samiliarity(attn_F,attn_B,attn_U) 
        
        out_h=self.forward_high_frequency(attn_F,attn_B,attn_U)
        
        out_l=self.forward_low_frequency(attn_F,attn_B,attn_U)
        
        out = out_s+out_h+out_l+x
        out = self.output_cbr(out)
        
        return out



class FinalPredictionHead(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        
        self.x1_pred = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channel[0], 1, kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x1):
        
        mask1=self.x1_pred(x1) 
        
        return mask1



class FeatureDecouplingUnit(nn.Module):
    def __init__(self, in_channel=1024,out_channels=1024,up_factor=1,channel_factor=[1,2,4],channel_scale=2,attention=nn.Sigmoid):
        super().__init__()
        
        self.fdu = FeatureDecoupling(in_channel=in_channel,out_channels=out_channels,channel_scale=channel_scale,attention=attention)
        
        self.fdu_ph = FeatureDecouplingPredictionHead(in_channel,up_factor=up_factor,channel_factor=channel_factor,channel_scale=channel_scale)
        
    def forward(self,x):
        f_fg, f_bg, f_uc = self.fdu(x)
        
        mask_fg, mask_bg, mask_uc = self.fdu_ph(f_fg, f_bg, f_uc)
        
        return f_fg, f_bg, f_uc,mask_fg, mask_bg, mask_uc


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2  

    def forward(self, image):

        x0 = image
        x1 = self.layer0(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)

        return x1, x2, x3



class CFAU_Preprocess(nn.Module):
    def __init__(self, in_channel,mid_channel, out_channel,channel_scale=2):
        super().__init__()
     
        self.preprocess = nn.Sequential(
            CBR(in_channel, in_channel//channel_scale, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(in_channel//channel_scale, mid_channel, kernel_size=3, padding=1),
        )
        
        self.pro=nn.Sequential(
            CBR(mid_channel*2, out_channel, kernel_size=3, padding=1),
            CBR(out_channel, out_channel, kernel_size=3, padding=1),
        )

    def forward(self, x1,x2):
        
        x1 = self.preprocess(x1)
        x=torch.cat([x1,x2],dim=1)
        x=self.pro(x)
        
        return x
    
    

class DecoSeg(nn.Module):
    def __init__(self, in_channels=[64, 256, 512],
                 up_factor=[1, 2, 3], channel_factor=[1, 2, 4], channel_scale=2, attention=nn.Sigmoid,
                 pos=DS_POS, attn_drop=0., proj_drop=0., num_filters=3, l_kernel=(3, 3), v_kernel=(5, 1),
                 h_kernel=(5, 1), p_kernel=[2, 4]):
        super().__init__()

        self.backbone = Backbone()

        self.fdu = nn.ModuleList()
        self.cfau = nn.ModuleList()

        self.cfau_pre = nn.ModuleList([
            CFAU_Preprocess(in_channels[2], in_channels[1], in_channels[1], channel_scale=channel_scale),  
            CFAU_Preprocess(in_channels[1], in_channels[0], in_channels[0], channel_scale=channel_scale),  
        ])

        for i in range(3):
            self.fdu.append(FeatureDecouplingUnit(in_channel=in_channels[i], out_channels=in_channels[i],
                                                  up_factor=up_factor[i], channel_factor=channel_factor,
                                                  channel_scale=channel_scale, attention=attention))
            self.cfau.append(ContrastDrivenFeatureAlignment(in_channel=in_channels[i], out_channel=in_channels[i],
                                                            channel_scale=channel_scale, pos=pos, attn_drop=attn_drop,
                                                            proj_drop=proj_drop, num_filters=num_filters,
                                                            l_kernel=l_kernel, v_kernel=v_kernel,
                                                            h_kernel=h_kernel, p_kernel=p_kernel))

        self.FPH = FinalPredictionHead(in_channels)

    def forward(self, image):
        mask_fg = []
        mask_bg = []
        mask_uc = []

        x1, x2, x3 = self.backbone(image)

        f_fg3, f_bg3, f_uc3, m_fg3, m_bg3, m_uc3 = self.fdu[2](x3)
        mask_fg.append(m_fg3)
        mask_bg.append(m_bg3)
        mask_uc.append(m_uc3)
        cin2 = self.cfau[2](x3, f_fg3, f_bg3, f_uc3)

        cin2 = self.cfau_pre[0](cin2, x2)
        f_fg2, f_bg2, f_uc2, m_fg2, m_bg2, m_uc2 = self.fdu[1](cin2)
        mask_fg.append(m_fg2)
        mask_bg.append(m_bg2)
        mask_uc.append(m_uc2)
        cin1 = self.cfau[1](cin2, f_fg2, f_bg2, f_uc2)

        cin1 = self.cfau_pre[1](cin1, x1)
        f_fg1, f_bg1, f_uc1, m_fg1, m_bg1, m_uc1 = self.fdu[0](cin1)
        mask_fg.append(m_fg1)
        mask_bg.append(m_bg1)
        mask_uc.append(m_uc1)
        cin1 = self.cfau[0](cin1, f_fg1, f_bg1, f_uc1)

        mask1 = self.FPH(cin1)
        return mask1, mask_fg, mask_bg, mask_uc
