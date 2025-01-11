from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from models.encoder_ae import encoder8x16x16 as sensitivity_encoder
from models.encoder_cls import encoder8x16x16 as insensitivity_encoder
from torch.autograd import Variable

'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR',
         negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                          negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C' + mode,
               negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                    negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R',
                           negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R',
                          negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R',
                       negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                       negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)


class SentivityAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 negative_slope=0.2):
        super(SentivityAttention, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x, gamma=None, beta=None):
        if gamma is not None:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            res = (gamma) * self.res(x) + beta
            return x + res
        else:
            return self.res(x)


class SensitivityEncoder(nn.Module):
    def __init__(self, in_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', ):
        super(SensitivityEncoder, self).__init__()
        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')
        self.nb = nb
        self.nc = nc
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode='2'))
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=True, mode='2'))

        self.m_body_encoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

    def forward(self, x):
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        return x


class SensitivityDecoder(nn.Module):
    def __init__(self, nc=[64, 128, 256, 512], nb=4, act_mode='R'):
        super(SensitivityDecoder, self).__init__()
        self.qf_pred = sequential(*[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                                  torch.nn.AdaptiveAvgPool2d((1, 1)),
                                  torch.nn.Flatten(),
                                  # nn.Sigmoid()
                                  )
        self.linear = sequential(torch.nn.Linear(512, 512),
                                 nn.ReLU(),
                                 torch.nn.Linear(512, 512),
                                 nn.ReLU(),
                                 torch.nn.Linear(512, 1)
                                 )

    def forward(self, x):
        f = self.qf_pred(x)
        x = self.linear(f)
        return x, f


class InsensitivityEncoder(nn.Module):
    def __init__(self, in_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv'):
        super(InsensitivityEncoder, self).__init__()
        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')
        self.nb = nb
        self.nc = nc
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode='2'))
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=True, mode='2'))

        self.m_body_encoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

    def forward(self, x):
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        return x


class InsensitivityDecoder(nn.Module):
    def __init__(self, nc=[64, 128, 256, 512], nb=4, act_mode='R'):
        super(InsensitivityDecoder, self).__init__()
        self.qf_pred = sequential(*[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                                  torch.nn.AdaptiveAvgPool2d((1, 1)),
                                  torch.nn.Flatten(),
                                  # nn.Sigmoid()
                                  )

    def forward(self, x):
        x = self.qf_pred(x)
        return x


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, low_in_channels, high_in_channels, key_channels, value_channels, out_channels=None, scale=1,
                 norm_type=None, psp_size=(1, 3, 6, 8)):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = low_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = high_in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=high_in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, low_feats, high_feats):
        batch_size, h, w = high_feats.size(0), high_feats.size(2), high_feats.size(3)
        # if self.scale > 1:
        #     x = self.pool(x)

        value = self.psp(self.f_value(low_feats))

        query = self.f_query(high_feats).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(low_feats)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *high_feats.size()[2:])
        context = self.W(context)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, low_in_channels, high_in_channels, key_channels, value_channels, out_channels=None, scale=1,
                 norm_type=None, psp_size=(1, 3, 6, 8)):
        super(SelfAttentionBlock2D, self).__init__(low_in_channels,
                                                   high_in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale,
                                                   norm_type,
                                                   psp_size=psp_size
                                                   )


class AFNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout=0.05,
                 sizes=([1]), norm_type=None, psp_size=(1, 3, 6, 8)):
        super(AFNB, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size = psp_size
        self.stages = nn.ModuleList(
            [self._make_stage([low_in_channels, high_in_channels], out_channels, key_channels, value_channels, size) for
             size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_channels + high_in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels[0],
                                    in_channels[1],
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_type,
                                    psp_size=self.psp_size)

    def forward(self, low_feats, high_feats):
        priors = [stage(low_feats, high_feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, high_feats], 1))
        return output


class AFNB2(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout=0.05,
                 sizes=([1]), norm_type=None, psp_size=(1, 3, 6, 8)):
        super(AFNB2, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size = psp_size
        self.stages = nn.ModuleList(
            [self._make_stage([low_in_channels, high_in_channels], out_channels, key_channels, value_channels, size) for
             size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_channels + high_in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels[0],
                                    in_channels[1],
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_type,
                                    psp_size=self.psp_size)

    def forward(self, low_feats, high_feats):
        priors = [stage(low_feats, high_feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = context + high_feats
        return output


class SimFusion(nn.Module):
    def __init__(self, channel):
        super(SimFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2 * channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channel)

    def forward(self, x, y):
        c = torch.cat([x, y], dim=1)
        c = F.relu(self.bn1(self.conv1(c)), inplace=True)
        c = F.relu(self.bn2(self.conv2(c)), inplace=True)
        c = c + x
        return c


class DFGN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, is_nb=2, act_mode='R',
                 downsample_mode='strideconv',
                 upsample_mode='convtranspose', fusion_mode='nonlocal', split_se=False, freeze_encoder=False, s_control=True, i_control=True):
        super(DFGN, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')
        self.nb = nb
        self.nc = nc
        # sensitivity
        self.se = sensitivity_encoder(in_nc=in_nc)

        # insensitivity
        self.ie = insensitivity_encoder(in_nc=in_nc)

        self.freeze_encoder = freeze_encoder
        # 0528
        if self.freeze_encoder:
            for k, v in self.se.named_parameters():
                v.requires_grad = False
            for k, v in self.ie.named_parameters():
                v.requires_grad = False

        # fusion
        self.fusion_mode = fusion_mode
        if self.fusion_mode == 'nonlocal':
            self.fusion_block = AFNB(2048, 512, 512, 256, 256, psp_size=[1,2,4,8])
        elif self.fusion_mode == 'nonlocal2':
            self.fusion_block = AFNB2(2048, 512, 512, 256, 256, psp_size=[1,2,4,8])
        elif self.fusion_mode == 'simple':
            self.fusion_block = SimFusion(512)
        elif self.fusion_mode == 'non':
            self.fusion_block = None
        else:
            raise NotImplementedError('fusion mode [{:s}] is not found'.format(fusion_mode))

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode='2'))
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=True, mode='2'))

        self.m_body_encoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_body_decoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = nn.ModuleList([upsample_block(nc[3], nc[2], bias=True, mode='2'),
                                    *[SentivityAttention(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in
                                      range(nb)],
                                    *[SentivityAttention(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in
                                      range(nb)]])

        self.m_up2 = nn.ModuleList([upsample_block(nc[2], nc[1], bias=True, mode='2'),
                                    *[SentivityAttention(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in
                                      range(nb)],
                                    *[SentivityAttention(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in
                                      range(nb)]])

        self.m_up1 = nn.ModuleList([upsample_block(nc[1], nc[0], bias=True, mode='2'),
                                    *[SentivityAttention(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in
                                      range(nb)],
                                    *[SentivityAttention(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in
                                      range(nb)]])

        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')

        self.qf_embed = sequential(torch.nn.Linear(2048, 512),
                                   nn.ReLU(),
                                   # torch.nn.Linear(512, 512),
                                   # nn.ReLU(),
                                   torch.nn.Linear(512, 512),
                                   nn.ReLU()
                                   )
        self.is_embed = sequential(torch.nn.Linear(2048, 512),
                                   nn.ReLU(),
                                   # torch.nn.Linear(512, 512),
                                   # nn.ReLU(),
                                   torch.nn.Linear(512, 512),
                                   nn.ReLU()
                                   )

        self.to_gamma_3_i = sequential(torch.nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3_i = sequential(torch.nn.Linear(512, nc[2]), nn.Tanh())
        self.to_gamma_2_i = sequential(torch.nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2_i = sequential(torch.nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_1_i = sequential(torch.nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1_i = sequential(torch.nn.Linear(512, nc[0]), nn.Tanh())

        self.to_gamma_3_s = sequential(torch.nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3_s = sequential(torch.nn.Linear(512, nc[2]), nn.Tanh())
        self.to_gamma_2_s = sequential(torch.nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2_s = sequential(torch.nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_1_s = sequential(torch.nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1_s = sequential(torch.nn.Linear(512, nc[0]), nn.Tanh())

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.s_control = s_control
        self.i_control = i_control
        print('sssssssss s_control :{}'.format(self.s_control))

    def forward(self, x, y=None, qf_input=None):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 16) * 16 - h)
        paddingRight = int(np.ceil(w / 16) * 16 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        if y is not None:
            y = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(y)

        # sensitivity and insensitivity
        # 0526
        if self.freeze_encoder:
            # self.se.eval()
            # self.ie.eval()
            with torch.no_grad():
                out_se, out_se_f = self.se(x)
                out_ie, out_ie_f = self.ie(x)

                out_se_f = Variable(out_se_f.detach_())

                out_ie_f = Variable(out_ie_f.detach_())
        else:
            out_se, out_se_f = self.se(x)
            # 0507
            out_ie, out_ie_f = self.ie(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        # mali
        # out_ie_f H//16, W//16; x H//8, W//8
        if self.fusion_mode == 'nonlocal' or self.fusion_mode == 'nonlocal2':
            x = self.fusion_block(out_ie_f, x)
        elif self.fusion_mode == 'simple':
            x = self.fusion_block(x, torch.nn.functional.pixel_shuffle(out_ie_f, 2))
        elif self.fusion_mode == 'non':
            pass
        else:
            raise NotImplementedError('fusion mode [{:s}] is not found'.format(self.fusion_mode))

        x = self.m_body_decoder(x)

        qf_embedding = self.qf_embed(torch.flatten(self.pooling(out_se_f), start_dim=1))
        is_embedding = self.is_embed(torch.flatten(self.pooling(out_ie_f), start_dim=1))
        gamma_3_s = self.to_gamma_3_s(qf_embedding)
        beta_3_s = self.to_beta_3_s(qf_embedding)

        gamma_2_s = self.to_gamma_2_s(qf_embedding)
        beta_2_s = self.to_beta_2_s(qf_embedding)

        gamma_1_s = self.to_gamma_1_s(qf_embedding)
        beta_1_s = self.to_beta_1_s(qf_embedding)

        gamma_3_i = self.to_gamma_3_i(is_embedding)
        beta_3_i = self.to_beta_3_i(is_embedding)

        gamma_2_i = self.to_gamma_2_i(is_embedding)
        beta_2_i = self.to_beta_2_i(is_embedding)

        gamma_1_i = self.to_gamma_1_i(is_embedding)
        beta_1_i = self.to_beta_1_i(is_embedding)

        x = x + x4
        x = self.m_up3[0](x)
        for i in range(self.nb):
            if self.s_control:
                x = self.m_up3[i + 1](x, gamma_3_s, beta_3_s)
            else:
                x = self.m_up3[i + 1](x, None)
        for i in range(self.nb):
            if self.i_control:
                x = self.m_up3[self.nb + i + 1](x, gamma_3_i, beta_3_i)
            else:
                x = self.m_up3[self.nb + i + 1](x, None)
        x = x + x3

        x = self.m_up2[0](x)
        for i in range(self.nb):
            if self.s_control:
                x = self.m_up2[i + 1](x, gamma_2_s, beta_2_s)
            else:
                x = self.m_up2[i + 1](x, None)
        for i in range(self.nb):
            if self.i_control:
                x = self.m_up2[self.nb + i + 1](x, gamma_2_i, beta_2_i)
            else:
                x = self.m_up2[self.nb + i + 1](x, None)
        x = x + x2

        x = self.m_up1[0](x)
        for i in range(self.nb):
            if self.s_control:
                x = self.m_up1[i + 1](x, gamma_1_s, beta_1_s)
            else:
                x = self.m_up1[i + 1](x, None)
        for i in range(self.nb):
            if self.i_control:
                x = self.m_up1[self.nb + i + 1](x, gamma_1_i, beta_1_i)
            else:
                x = self.m_up1[self.nb + i + 1](x, None)

        x = x + x1
        x = self.m_tail(x)
        pred = x[..., :h, :w]

        return pred, out_ie_f, out_se_f, None, None, None
