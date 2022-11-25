# --------------------------------------------------------
# EVA: Exploring the Limits of Masked Visual Representation Learning at Scale (https://arxiv.org/abs/2211.07636)
# Github source: https://github.com/baaivision/EVA
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
# Based on timm, DINO, DeiT and BEiT codebases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------'

import clip
import torch.nn as nn


class CLIPWrapper(nn.Module):
    def __init__(self, clip_model="ViT-L/16", download_root=None):
        super().__init__()
        self.net, _ = clip.load(
            clip_model, device='cpu', jit=False, 
            download_root=download_root, 
        )

    def infer_image(self, features):
        x = features["image"][0]
        x = self.net.encode_image(x)
        return x

