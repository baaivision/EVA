from functools import partial

from ..common.coco_loader_lsj_1408 import dataloader
from .cascade_mask_rcnn_vitdet_b_100ep import (
    # dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = "/sharefs/baaivision/xinlongwang/models/mae/" \
                        "mae_vit_giant_patch14_150ep_8x8gpu_in21k_70ep_bf16/checkpoint-149-s14tos16.pth"

model.backbone.net.img_size = 1280  # 1024
model.backbone.square_pad = 1408  # 1024
model.backbone.net.patch_size = 16  # 14 --> 16
model.backbone.net.window_size = 16  # 14 --> 16
model.backbone.net.embed_dim = 1408
model.backbone.net.depth = 40
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 6144 / 1408
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.6  # 0.5 --> 0.6
# 7, 15, 23, 31 for global attention
model.backbone.net.window_block_indexes = (
    # list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31)) + list(range(32, 39))
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
    list(range(20, 23)) + list(range(24, 27)) + list(range(28, 31)) + list(range(32, 35)) + list(range(36, 39))
    # list(range(0, 40))
)
# model.backbone.net.residual_block_indexes = (
#     list(range(3, 41, 4))
# )

optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.9, num_layers=40)  # 32 --> 40
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

train.max_iter = train.max_iter * 3 // 4  # 100ep -> 75ep
lr_multiplier.scheduler.milestones = [
    train.max_iter-2, train.max_iter-1,
#    milestone * 3 // 4 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 0 / train.max_iter  # 2ep 118k*2/64
