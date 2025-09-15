# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import warnings

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
                  
    return F.interpolate(input, size, scale_factor, mode, align_corners)



def kaiming_init(module: nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.
    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.
    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = 'auto',
                 conv_cfg = None,
                 norm_cfg = None,
                 act_cfg = dict(type='ReLU'),
                 inplace= True,
                 with_spectral_norm = False,
                 padding_mode = 'zeros',
                 order = ('conv', 'norm', 'act')):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)
        # to do Camille put back

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = nn.Conv2d(  #build_conv_layer(#conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        self.norm_name = None  # type: ignore

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = nn.ReLU() # build_activation_layer(act_cfg_)

        # Use msra init by default
        torch.manual_seed(1)
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self,
                x: torch.Tensor,
                activate: bool = True,
                norm: bool = True, 
                debug: bool = False) -> torch.Tensor:
        
        for layer in self.order:
            if debug==True:
                breakpoint()
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        return x



class ReassembleBlocks(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """
    def __init__(self,
                 in_channels=1024, #768,
                 out_channels=[128, 256, 512, 1024],  #[96, 192, 384, 768],
                 readout_type='project', # 'ignore',
                 patch_size=16):
        super(ReassembleBlocks, self).__init__()#init_cfg)

        assert readout_type in ['ignore', 'add', 'project']
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList([
            ConvModule(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                act_cfg=None,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        if self.readout_type == 'project':
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(6 * in_channels, in_channels), #4 register tokens + 1 cls token 
                        nn.GELU()))
                        #build_activation_layer(dict(type='GELU'))))

    def forward(self, inputs):
        """
        inputs: list of (feat, regs, cls)
        - feat: [B, C, H, W]
        - regs: [B, R, C]  (R>=1)
        - cls : [B, C]
        readout_type: 'project'
        - self.readout_projects[i]: Linear((2+R)*C -> C)
        - self.projects[i], self.resize_layers[i]
        """
        out = []
        for i, (x, regs, cls_token) in enumerate(inputs):

            assert x.dim() == 4, "feat must be [B,C,H,W]"
            assert regs is not None and regs.dim() == 3 and regs.size(1) >= 1, "regs must be [B,R,C] with R>=1"
            assert cls_token.dim() == 2, "cls must be [B,C]"
            B, C, H, W = x.shape
            R = regs.size(1)


            readout_tokens = torch.cat([cls_token.unsqueeze(1), regs], dim=1)   # [B, 1+R, C]
            readout_tokens = F.layer_norm(readout_tokens, readout_tokens.shape[-1:])

            x_flat = x.flatten(2).transpose(1, 2)                               # [B, H*W, C]

            readout_vec = readout_tokens.reshape(B, (1 + R) * C)                # [B, (1+R)C]
            readout_bc  = readout_vec.unsqueeze(1).expand(B, x_flat.size(1), -1)# [B, H*W, (1+R)C]

            # -> [B, H*W, (2+R)C] ----
            x_cat = torch.cat([x_flat, readout_bc], dim=-1)

            x_proj = self.readout_projects[i](x_cat)                             # Linear((2+R)*C -> C)
            x = x_proj.transpose(1, 2).reshape(B, C, H, W)

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        return out


class PreActResidualConvUnit(nn.Module):
    """ResidualConvUnit, pre-activate residual unit.
    Args:
        in_channels (int): number of channels in the input feature map.
        act_cfg (dict): dictionary to construct and config activation layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        stride (int): stride of the first block. Default: 1
        dilation (int): dilation rate for convs layers. Default: 1.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self,
                 in_channels,
                 act_cfg,
                 norm_cfg,
                 stride=1,
                 dilation=1,
                 init_cfg=None):
        super(PreActResidualConvUnit, self).__init__()#init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=('act', 'conv', 'norm'))
        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=('act', 'conv', 'norm'))
    def forward(self, inputs):
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_


class FeatureFusionBlock(nn.Module):
    """FeatureFusionBlock, merge feature map from different stages.
    Args:
        in_channels (int): Input channels.
        act_cfg (dict): The activation config for ResidualConvUnit.
        norm_cfg (dict): Config dict for normalization layer.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self,
                 in_channels,
                 act_cfg,
                 norm_cfg,
                 expand=False,
                 align_corners=True,
                 init_cfg=None):
        super(FeatureFusionBlock, self).__init__()#init_cfg)
        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners
        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2
        self.project = ConvModule(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            act_cfg=None,
            bias=True)
        self.res_conv_unit1 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.res_conv_unit2 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)

    def forward(self, *inputs):
        x = inputs[0] 
        
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = resize(
                    inputs[1],
                    size=(x.shape[2], x.shape[3]),
                    mode='bilinear',
                    align_corners=False)
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x) 
        x = resize( x, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        x = self.project(x) 
        return x


class HeadDepth(nn.Module):
    def __init__(self, features, classify=False, n_bins=256):
        super(HeadDepth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1 if not classify else n_bins, kernel_size=1, stride=1, padding=0),
        )
    def forward(self, x):
        x = self.head(x)
        return x    

class HeadSeg(nn.Module):
    def __init__(self, features, n_output_channels, dropout_ratio=0.1, use_sync_bn=False):
        super(HeadSeg, self).__init__()
        Norm = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            Norm(features),
            nn.ReLU(True),
            nn.Dropout2d(dropout_ratio, inplace=False),
            nn.Conv2d(features, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)
    
class DPTHead(nn.Module):
    """
    Produces per-pixel logits for semantic segmentation.
    Expects ViT backbone features: list of 4 entries, each (feat, cls_token).
    """
    def __init__(self,
                 in_channels=(1024, 1024, 1024, 1024),
                 channels=256,
                 embed_dims=1024,
                 post_process_channels=[128, 256, 512, 1024],
                 readout_type='project',
                 patch_size=16,
                 expand_channels=False,
                 seg_cls_number=6,
                 upsample_mode='bilinear',
                 use_sync_bn=False):
        super().__init__()
        self.channels = channels
        self.norm_cfg = None
        self.in_channels = in_channels
        self.expand_channels = expand_channels
        self.seg_cls_number = seg_cls_number
        self.upsample_mode = upsample_mode

        self.reassemble_blocks = ReassembleBlocks(
            in_channels=embed_dims,
            out_channels=post_process_channels,
            readout_type=readout_type,
            patch_size=patch_size
        )

        self.post_process_channels = [
            int(channel * math.pow(2, i)) if expand_channels else int(channel)
            for i, channel in enumerate(post_process_channels)
        ]

        self.convs = nn.ModuleList([
            ConvModule(c, self.channels, kernel_size=3, padding=1, act_cfg=None, bias=False)
            for c in self.post_process_channels
        ])

        self.act_cfg = {'type': 'ReLU'}
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(self.channels, self.act_cfg, self.norm_cfg)
            for _ in range(len(self.convs))
        ])
        # First fusion block does not use the skip’s res unit1 (DPT convention)
        self.fusion_blocks[0].res_conv_unit1 = None

        self.project = ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm_cfg=None)

        # Final classifier
        self.conv_seg = HeadSeg(self.channels, self.seg_cls_number, dropout_ratio=0.1, use_sync_bn=use_sync_bn)

        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks == self.num_post_process_channels

    def forward(self, inputs):
        # inputs: list of 4 tuples (feat_map, cls_token)
        x = self.reassemble_blocks(inputs)                       # 4 scales, CNN-ized
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        out = self.project(out)
        out = resize(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv_seg(out)                 # [B, num_classes, h, w]
        # if img_size_hw is not None and (logits.shape[-2] != img_size_hw[0] or logits.shape[-1] != img_size_hw[1]):
        #     logits = F.interpolate(logits, size=img_size_hw, mode=self.upsample_mode, align_corners=False)
        return out
