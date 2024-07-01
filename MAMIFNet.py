import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import sys
from mmcv.cnn import build_norm_layer
sys.path.insert(0, '../../')
from pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b2_li, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5

import numpy as np
import cv2

from MAMFFC import MAM_FFC_Layer

class MAMIFNet(nn.Module):
    def __init__(self, fun_str='pvt_v2_b4',dim=64):
        super().__init__()
        self.backbone, embedding_dims = eval(fun_str)()

        self.dem4 = DEM(dim)
        self.dem3 = DEM(dim)
        self.dem2 = DEM(dim)
        self.dem1 = DEM(dim)

        self.fu0 = fusion2(cur_in_channels=embedding_dims[0], low_in_channels=embedding_dims[1],
                                  out_channels=dim, cur_scale=1, low_scale=2)
        self.fu1 = fusion2(cur_in_channels=embedding_dims[1], low_in_channels=embedding_dims[2],
                        out_channels=dim, cur_scale=1, low_scale=2)
        self.fu2 = fusion2(cur_in_channels=embedding_dims[2], low_in_channels=embedding_dims[3],
                                  out_channels=dim, cur_scale=1, low_scale=2)
        self.fu3 = fusion2(cur_in_channels=dim, low_in_channels=dim,
                                             out_channels=dim, cur_scale=2,
                                             low_scale=8)  # 16
        self.fu4 = fusion3(cur_in_channels=320, low_in_channels=128,dep_in_channels=512,
                                             out_channels=dim)  # 16

        self.pipm5 = PIPM5(in_channels=dim, out_channels=dim,
                                          kernel_size=3, stride=1,
                                          dilation=1, padding=1)

        self.predict_conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, padding=1, stride=1))




    def forward(self, x):
        # byxhz
        layer = self.backbone(x)
        shape = x.shape[-2:]


        # P5, P4, P3, P2, P1 = self.decoder(layer[3], layer[2], layer[1], layer[0],shape)
        # layer0 b,64,96,96
        # layer1 b,128,48,48
        # layer2 b,320,24,24
        # layer3 b,512,12,12

        s2 = self.fu0(layer[0], layer[1])   #s2 1,64,96,96

        s3 = self.fu1(layer[1], layer[2])  #s3 1,64,48,48

        s4 = self.fu2(layer[2], layer[3]) #s4 1,64,24,24

        s5 = self.fu4( layer[1],layer[2],layer[3])  #s5 1,64,24,24

        s1 = self.fu3(s2, s5)   #s1 1,64,192,192

        t5 = self.pipm5(s5)     #t5 1,64,24,24

        predict5 = self.predict_conv(t5)  #predict5 1,64,24,24

        # focus
        dem4, predict4 = self.dem4(s4, s5, t5, predict5)

        dem3, predict3 = self.dem3(s3,s4, dem4, predict4)

        dem2, predict2 = self.dem2(s2,s3, dem3, predict3)

        dem1, predict1 = self.dem1(s1,s2, dem2, predict2)

        # rescale
        predict5 = F.interpolate(predict5, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)

        return  predict5, predict4, predict3, predict2, predict1
        # return P5, P4, P3, P2, P1

class PIPM5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(PIPM5, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)
        self.pipm  = MAM_FFC_Layer(out_channels, ratio_gin=0.5,
                 ratio_gout=0.5,fre_m=0.0)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pipm(x)

        return x

class fusion2(nn.Module):
    def __init__(self, cur_in_channels=64, low_in_channels=32, out_channels=64, cur_scale=2, low_scale=1):
        super(fusion2, self).__init__()
        self.cur_in_channels = cur_in_channels
        self.cur_conv = nn.Sequential(
            nn.Conv2d(in_channels=cur_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),

        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),

        )

        self.cur_scale = cur_scale
        self.low_scale = low_scale

        self.out_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2 * out_channels, out_channels= 1 * out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1 * out_channels),
        )


    def forward(self, x_cur, x_low):
        x_cur = self.cur_conv(x_cur)
        # bicubic bilinear nearest
        x_cur = F.interpolate(x_cur, scale_factor=self.cur_scale, mode='bicubic', align_corners=False)

        x_low = self.low_conv(x_low)
        x_low = F.interpolate(x_low, scale_factor=self.low_scale, mode='bicubic', align_corners=False)
        x = torch.cat((x_cur, x_low), dim=1)
        x = self.out_conv1(x)

        return x

class fusion3(nn.Module):
    def __init__(self, dep_in_channels=32,cur_in_channels=64, low_in_channels=32, out_channels=16, cur_scale=2, low_scale=1):
        super(fusion3, self).__init__()
        self.cur_in_channels = cur_in_channels
        self.cur_conv = nn.Sequential(
            nn.Conv2d(in_channels=cur_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.dep_conv = nn.Sequential(
            nn.Conv2d(in_channels=dep_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.conv1 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)

        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)


    def forward(self,x_low, x_cur,x_dep):
        x_low = F.interpolate(x_low, size=x_cur.size()[2:], mode='bilinear', align_corners=True)
        x_dep = F.interpolate(x_dep, size=x_cur.size()[2:], mode='bilinear', align_corners=True)

        x_cur = self.cur_conv(x_cur)
        x_low = self.low_conv(x_low)
        x_dep = self.dep_conv(x_dep)


        x = torch.cat((x_cur, x_low,x_dep), dim=1)
        fuse = self.bn1(self.conv1(x))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse



def get_open_map(input, kernel_size, iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations),
                        input.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()


class Basic_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DEM(nn.Module):
    def __init__(self, channel):
        super(DEM, self).__init__()
        self.channel = channel
        # self.channel2 = channel

        self.up = nn.Sequential(nn.Conv2d(self.channel, self.channel, 7, 1, 3),
                                nn.BatchNorm2d(self.channel), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        # ???????
        self.increase_input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        self.output_map = nn.Conv2d(self.channel, 1, 7, 1, 3)
        self.beta1 = nn.Parameter(torch.ones(1))
        self.beta2 = nn.Parameter(torch.ones(1))

        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1,
                               stride=1)

        self.conv_cur_dep1 = Basic_Conv(self.channel, self.channel, 3, 1, 1)

        self.conv_cur_dep2 = Basic_Conv(in_channels=self.channel, out_channels=self.channel, kernel_size=3,
                                        padding=1, stride=1)

        self.conv_cur_dep3 = Basic_Conv(in_channels=self.channel, out_channels=self.channel, kernel_size=3,
                                        padding=1, stride=1)
        self.pipm  = MAM_FFC_Layer(self.channel, ratio_gin=0.5,
                 ratio_gout=0.5,fre_m=0.0)
        self.arem  = MAM_FFC_Layer(self.channel, ratio_gin=0.5,
                 ratio_gout=0.5,fre_m=0.25)
    def cgpm(self, dep_x_dual, dep_mask, input_map):

        t_dep_feature = dep_x_dual

        fc_dep_feature = self.arem(dep_mask, input_map)

        dep_feature = self.beta2 * fc_dep_feature + t_dep_feature

        return dep_feature
    def forward(self, cur_x_spa,dep_x_spa, dep_x_dual, in_map):
        # x; current-level features    1,64,24,24
        # y: higher-level features     1,64,12,12
        # in_map: higher-level prediction  ,1,1,12,12
        if dep_x_dual.shape[3:]!=cur_x_spa.shape[3:]:

            dep_x_dual = self.up(dep_x_dual)
        if dep_x_spa.shape[3:]!=cur_x_spa.shape[3:]:

            dep_x_spa = self.up(dep_x_spa)
        if in_map.shape[3:] != cur_x_spa.shape[3:]:
            # dep_x_dual = self.up(dep_x_dual)
            input_map = self.input_map(in_map)
        else:
            input_map = in_map

        dep_mask= dep_x_spa * input_map  # ????????????????????????????????
        # cur_mask = cur_x_spa * input_map  # ????????????????????????????????
        # dep_x_dual   1,64,24,24
        # in_map: higher-level prediction  ,1,1,24,24

        dep_feature = self.cgpm(dep_x_dual, dep_mask, input_map)
        cur_feature = self.pipm(cur_x_spa)
        refine1 = dep_feature * cur_feature + cur_feature
        refine2 = self.conv_cur_dep1(refine1)
        refine2 = self.conv_cur_dep2(refine2)
        refine2 = self.conv_cur_dep3(refine2)
        output_map = self.output_map(refine2)

        return refine2, output_map


if __name__ == '__main__':
    import torch

    print(torch.__version__)
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from thop import profile

    net = MAMIFNet('pvt_v2_b4',dim=64).cuda()
    data = torch.randn(1, 3, 384, 384).cuda()
    # flops, params = profile(net, (data,))
    # print('flops: %.2f G, params: %.2f M' % (flops / (1024 * 1024 * 1024), params / (1024 * 1024)))
    y = net(data)
    # print(net)
    for i in y:
        print(i.shape)


