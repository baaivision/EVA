from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import torch.utils.checkpoint as checkpoint
import sys
from .utils import get_sinusoid_encoding_table

sys.path.append("../")
from clip.model import LayerNorm, DropPath, QuickGELU

from einops import rearrange
import numpy as np


class SpaceTimeAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath=0., init_values=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, )
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(d_model), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(d_model), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attention(self.ln_1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None,
                 init_values=0.,
                 use_checkpoint=False, checkpoint_segments=6):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.checkpoint_segments = checkpoint_segments
        if droppath is None:
            droppath = [0.0 for i in range(layers)]

        self.resblocks = nn.Sequential(
            *[SpaceTimeAttentionBlock(width, heads, attn_mask, droppath[i], init_values) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        if not self.use_checkpoint:
            return self.resblocks(x)
        else:
            return checkpoint_sequential(self.resblocks, self.checkpoint_segments, x)


class GlobalVideoTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 num_frames=8, tubelet_size=1, droppath=None, init_values=0., use_checkpoint=False, myclip_dict=None, num_classes=1000):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.width = width
        self.output_dim = output_dim
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.input_frame = num_frames // tubelet_size
        self.use_learnable_pos_emb = myclip_dict['USE_LEARNABLE_POS_EMB'] if myclip_dict is not None else False
        self.use_mean_pooling = myclip_dict['USE_MEAN_POOLING'] if myclip_dict is not None else False

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=width,
                               kernel_size=(self.tubelet_size, patch_size, patch_size),
                               stride=(self.tubelet_size, patch_size, patch_size), bias=False)  # VideoMAE bias=True

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.use_same_PE = myclip_dict['USE_SAME_PE'] if myclip_dict is not None else False
        self.use_global_atten = myclip_dict['USE_GLOBAL_ATTEN'] if myclip_dict is not None else True
        if self.use_same_PE:
            self.num_patches = (input_resolution // patch_size) ** 2
        else:
            self.num_patches = (input_resolution // patch_size) ** 2 * (num_frames // self.tubelet_size)
        # self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))

        if self.use_learnable_pos_emb:
            self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + 1, width))
        else:  # sine-cosine positional embeddings is on the way
            self.positional_embedding = get_sinusoid_encoding_table(self.num_patches + 1, width)

        self.ln_pre = LayerNorm(width)

        self.checkpoint_segments = myclip_dict['CHECKPOINT_SEGMENTS'] if myclip_dict is not None else 1
        self.transformer = Transformer(width, layers, heads, droppath=droppath, init_values=init_values,
                                       use_checkpoint=use_checkpoint, checkpoint_segments=self.checkpoint_segments)

        self.ln_post = nn.Identity() if self.use_mean_pooling else LayerNorm(width)
        self.fc_norm = LayerNorm(width) if self.use_mean_pooling else None

        self.proj = nn.Parameter(scale * torch.randn(width, output_dim if myclip_dict['USE_TEXT_EMBED'] else num_classes))

        trunc_normal_(self.proj, std=.02)

    def forward(self, x: torch.Tensor):

        b, t, c, h, w = x.size()

        x = x.transpose(1, 2)
        x = self.conv1(x).flatten(3)
        x = x.transpose(1, 2)

        if self.use_same_PE:
            x = rearrange(x, 'b t1 d l -> (b t1) l d')
        else:
            x = rearrange(x, 'b t1 d l -> b (t1 l) d')

        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)

        if self.use_learnable_pos_emb:
            x = x + self.positional_embedding.to(x.dtype)
        else:
            x = x + self.positional_embedding.type_as(x).to(x.device).clone().detach()

        x = self.ln_pre(x)
        if self.use_same_PE and self.use_global_atten:
            x = x.reshape(b, -1, x.shape[-1])

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.use_same_PE and self.use_global_atten:
            x = x.reshape(b * t // self.tubelet_size, -1, x.shape[-1])

        x = self.ln_post(x)

        if self.use_mean_pooling:
            cls_x = self.fc_norm(x[:, 1:, :].mean(1))
        else:
            cls_x = x[:, 0, :]

        if self.proj is not None:
            cls_x = cls_x @ self.proj

        if self.use_same_PE:
            cls_x = cls_x.reshape(b, -1, cls_x.shape[-1])
            cls_x = cls_x.mean(1)

        return cls_x
