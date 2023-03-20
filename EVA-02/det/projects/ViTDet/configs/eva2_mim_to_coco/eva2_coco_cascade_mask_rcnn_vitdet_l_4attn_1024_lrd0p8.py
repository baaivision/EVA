from functools import partial

from ..common.coco_loader_lsj_1024 import dataloader
from .cascade_mask_rcnn_vitdet_b_100ep import (
    # dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = ""

model.backbone.net.img_size = 1024 
model.backbone.square_pad = 1024  
model.backbone.net.patch_size = 16  
model.backbone.net.window_size = 16  
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = False
model.backbone.net.drop_path_rate = 0.4  

# 5, 11, 17, 23 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
)

optimizer.lr=6e-5
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None


train.max_iter = 60000
lr_multiplier.scheduler.milestones = [
    train.max_iter*8//10, train.max_iter*9//10
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter

dataloader.test.num_workers=0
dataloader.train.total_batch_size=144

