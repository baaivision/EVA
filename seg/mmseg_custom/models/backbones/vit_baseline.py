# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from timm.models.layers import trunc_normal_

from .base.vit import TIMMVisionTransformer

_logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class ViTBaseline(TIMMVisionTransformer):
    def __init__(self, pretrain_size=224, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.flags = [i for i in range(-1, self.num_block, self.num_block // 4)][1:]

        embed_dim = self.embed_dim
        self.norm1 = self.norm_layer(embed_dim)
        self.norm2 = self.norm_layer(embed_dim)
        self.norm3 = self.norm_layer(embed_dim)
        self.norm4 = self.norm_layer(embed_dim)

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        ])
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward_features(self, x):
        outs = []
        x, H, W = self.patch_embed(x)
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        for index, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            if index in self.flags:
                outs.append(x)
        return outs, H, W

    def forward(self, x):
        outs, H, W = self.forward_features(x)
        f1, f2, f3, f4 = outs
        bs, n, dim = f1.shape
        f1 = self.norm1(f1).transpose(1, 2).reshape(bs, dim, H, W)
        f2 = self.norm2(f2).transpose(1, 2).reshape(bs, dim, H, W)
        f3 = self.norm3(f3).transpose(1, 2).reshape(bs, dim, H, W)
        f4 = self.norm4(f4).transpose(1, 2).reshape(bs, dim, H, W)

        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()

        return [f1, f2, f3, f4]
