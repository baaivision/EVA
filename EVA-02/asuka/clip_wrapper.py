# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on EVA: Exploring the Limits of Masked Visual Representation Learning at Scale (https://arxiv.org/abs/2211.07636)
# https://github.com/baaivision/EVA/tree/master/EVA-01
# --------------------------------------------------------'


import eva_clip
import torch.nn as nn


class EVACLIPWrapper(nn.Module):
    def __init__(self, clip_model='EVA_CLIP_g_14', cache_dir='/sharefs/baaivision/yxf/weights/eva_clip/eva_clip_psz14.pt'):
        super().__init__()
        self.net, _ = eva_clip.build_eva_model_and_transforms(
                                        clip_model, 
                                        pretrained=cache_dir)

    def infer_image(self, features):
        x = features["image"][0]
        x = self.net.encode_image(x)
        return x
