# Modified from https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import PIL

# from CBAM import CBAMLayer3 as CBAMLayer
class CBAMLayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAMLayer, self).__init__()

    def forward(self, x):

        return x


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)

    # def initialize(self):
    #     weight_init(self)

class MAM_FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=True, se_kwargs=None, ffc3d=False, fft_norm='ortho',fre_m=0.25):
        # bn_layer not used
        super(MAM_FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 ,
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.relu = torch.nn.ReLU(inplace=False)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.cbam= CBAMLayer(self.conv_layer.in_channels)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm
        self.frequency_m_rate = fre_m

    def forward(self, x,spa_mask=None):
        batch = x.shape[0]
        # print(x.device)

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)


        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.fftn(x, dim=fft_dim, norm=self.fft_norm)
        if spa_mask is not None:
            img_fft_shift = torch.fft.fftshift(ffted, dim=(-2, -1))
            h, w = img_fft_shift.shape[-2:]
            frequency_m = torch.zeros((h, w)).cuda()
            ch, cw = h // 2, w // 2
            line = int((w * h * self.frequency_m_rate) ** .5 // 2)
            # print( self.frequency_m_rate)
            frequency_m[cw - line:cw + line, ch - line:ch + line] = 1
            img_fft_shift_process = img_fft_shift * (1 - frequency_m) + frequency_m * (-10)  # ????
            img_fft_ishift = torch.fft.ifftshift(img_fft_shift_process)

        else:
            img_fft_ishift = ffted

        ffted = torch.stack((img_fft_ishift.real, img_fft_ishift.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.cbam(ffted)
        # ffted = torch.concat((img_fft_ishift.real, img_fft_ishift.imag), dim=1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.ifftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

    # def initialize(self):
    #     weight_init(self)

class MAM_SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, fre_m =0.25,**fu_kwargs):
        # bn_layer not used
        super(MAM_SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.ReLU(inplace=True)
        )
        self.mam_fu = MAM_FourierUnit(
            out_channels // 2, out_channels // 2, groups,fre_m =fre_m, **fu_kwargs)
        if self.enable_lfu:
            self.mam_lfu = MAM_FourierUnit(
                out_channels // 2, out_channels // 2, groups,fre_m =fre_m,)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x,spa_mask =None):

        x = self.downsample(x)

        x = self.conv1(x)
        output = self.mam_fu(x,spa_mask)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.mam_lfu(xs,spa_mask)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        x = x.float()
        output = output.float()
        xs = xs.float()

        output = self.conv2(x + output + xs)

        return output


class MAM_FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False,  fre_m =0.25,**spectral_kwargs):
        super(MAM_FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg



        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.cbam_l2l = CBAMLayer(out_cl)


        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.cbam_l2g = CBAMLayer(out_cg)


        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.cbam_g2l = CBAMLayer(out_cl)


        module = nn.Identity if in_cg == 0 or out_cg == 0 else MAM_SpectralTransform


        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu,  fre_m =fre_m,**spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x, spa_mask=None,fname=None):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if spa_mask is not None:

            x_l_fc = x_l * spa_mask
            x_g_fc = x_g * spa_mask
        else:
            x_l_fc = x_l
            x_g_fc = x_g
        if self.ratio_gout != 1:
            convl2l =self.cbam_l2l(self.convl2l(x_l_fc))
            convg2l =self.cbam_g2l(self.convg2l(x_g_fc))
            out_xl = convl2l + convg2l * g2l_gate

        if self.ratio_gout != 0:
            spec_x = self.convg2g(x_g, spa_mask)
            convl2g = self.cbam_l2g(self.convl2g(x_l_fc))
            out_xg = convl2g * l2g_gate + spec_x

        return out_xl, out_xg


class MAM_FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True,  fre_m =0.25,**kwargs):
        super(MAM_FFC_BN_ACT, self).__init__()
        self.mam_ffc = MAM_FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, fre_m =fre_m,**kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x,spa_mask =None, fname=None):
        x_l, x_g = self.mam_ffc(x, spa_mask,fname=fname)

        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g



class MAM_FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, ratio_gin=0.5, ratio_gout=0.5 ,fre_m =0.25,):
        super().__init__()
        self.conv1 = MAM_FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, fre_m =fre_m)
        self.inline = inline

    def forward(self, x, spa_mask =None,fname=None):

        if self.inline:
            x_l, x_g = x[:, :-self.conv1.mam_ffc.global_in_num], x[:, -self.conv1.mam_ffc.global_in_num:]
        else:
            # x_l, x_g = x if type(x) is tuple else (x, 0)
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g),spa_mask, fname=fname)


        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out




class MAM_FFC_Block(torch.nn.Module):
    def __init__(self,
                 dim,  # Number of output/input channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 padding,
                 ratio_gin=0.5,
                 ratio_gout=0.5,
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 fre_m=0.25,
                 ):
        super().__init__()
        if activation == 'linear':
            self.activation = nn.Identity
        else:
            self.activation = nn.ReLU
        self.padding = padding
        self.kernel_size = kernel_size
        self.mam_ffc_block = MAM_FFCResnetBlock(dim=dim,
                                        padding_type='reflect',
                                        norm_layer=nn.BatchNorm2d,
                                        activation_layer=self.activation,
                                        dilation=1,
                                        ratio_gin=ratio_gin,
                                        ratio_gout=ratio_gout, fre_m =fre_m)

        self.concat_layer = ConcatTupleLayer()

    def forward(self, gen_ft, spa_mask=None, fname=None):
        x = gen_ft.float()

        x_l, x_g = x[:, :-self.mam_ffc_block.conv1.mam_ffc.global_in_num], x[:, -self.mam_ffc_block.conv1.mam_ffc.global_in_num:]
        id_l, id_g = x_l, x_g
        input_x = (x_l, x_g)
        x_l, x_g = self.mam_ffc_block(input_x,spa_mask, fname=fname)
        x_l, x_g = id_l + x_l, id_g + x_g
        x = self.concat_layer((x_l, x_g))

        return x + gen_ft.float()




class MAM_FFC_Layer(torch.nn.Module):
    def __init__(self,
                 dim,  # Number of input/output channels.
                 kernel_size=3,  # Convolution kernel size.
                 ratio_gin=0.5,
                 ratio_gout=0.5,
                 fre_m=0.25,
                 ):
        super().__init__()
        self.padding = kernel_size // 2

        self.mam_ffc_act1 = MAM_FFC_Block(dim=dim, kernel_size=kernel_size, activation=nn.ReLU,
                                padding=self.padding, ratio_gin=ratio_gin, ratio_gout=ratio_gout, fre_m =fre_m)
        self.mam_ffc_act2 = MAM_FFC_Block(dim=dim, kernel_size=kernel_size, activation=nn.ReLU,
                                padding=self.padding, ratio_gin=ratio_gin, ratio_gout=ratio_gout, fre_m =fre_m)


    def forward(self, gen_ft, spa_mask=None, fname=None):
        x = self.mam_ffc_act1(gen_ft, spa_mask, fname=fname)
        x = self.mam_ffc_act2(x, spa_mask, fname=fname)
        return x


if __name__ == '__main__':
    input = torch.randn(1, 128, 384, 384).cuda()
    mask = torch.randn(1, 1, 384, 384)
    mask = torch.sigmoid(mask).cuda()
    fc = input * mask
    bg = input * (1 - mask)
    channels = 128
    net = MAM_FFC_Layer(dim =channels,fre_m=0.9).cuda()

    out2 = net(fc,mask)

    print(out2.shape)