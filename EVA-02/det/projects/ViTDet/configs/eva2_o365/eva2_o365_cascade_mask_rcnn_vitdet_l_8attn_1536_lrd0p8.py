from functools import partial

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import WarmupParamScheduler
from detectron2.config import LazyCall as L

from ..common.objects365_trainval_loader_lsj_1536 import dataloader
from .cascade_mask_rcnn_vitdet_b_100ep import (
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = ""

# for o365
model.roi_heads.mask_in_features = None
model.roi_heads.mask_pooler = None
model.roi_heads.mask_head = None
model.roi_heads.num_classes = 365

# for model
model.backbone.net.img_size = 1536  
model.backbone.square_pad = 1536  
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = False
model.backbone.net.drop_path_rate = 0.4

# 2, 5, 8, 11, 14, 17, 20, 23 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
)

optimizer.lr=6e-5
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

train.max_iter = 400000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[train.max_iter*8//10, train.max_iter*9//10],
        num_updates=train.max_iter,
    ),
    warmup_length=5000 / train.max_iter,
    warmup_factor=0.001,
)

dataloader.test.num_workers=0
dataloader.train.total_batch_size=160

train.checkpointer.period = 2500
train.eval_period = 10000

