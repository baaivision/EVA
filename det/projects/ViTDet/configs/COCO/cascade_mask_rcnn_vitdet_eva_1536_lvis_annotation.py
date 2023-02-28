from functools import partial

from ..common.coco_loader_lsj_1536 import dataloader
from .cascade_mask_rcnn_vitdet_b_100ep import (
    # dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)


from detectron2.config import LazyCall as L
from detectron2.evaluation.lvis_evaluation import LVISEvaluator

dataloader.test.dataset.names = "lvis_v1_val_cocofied"
dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)


dataloader.train.total_batch_size = 64

model.backbone.net.beit_like_qkv_bias = True
model.backbone.net.beit_like_gamma = False
model.backbone.net.freeze_patch_embed = True
model.backbone.square_pad = 1536
model.backbone.net.img_size = 1280  # only for correct dim in pos embed
model.backbone.net.interp_type = "beit"  # for eval, slightly AP improvement at a higher res, e.g., 1280 training --> 1536 eval 
model.backbone.net.patch_size = 16
model.backbone.net.window_size = 16
model.backbone.net.embed_dim = 1408
model.backbone.net.depth = 40
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 6144 / 1408
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.6  # 0.5 --> 0.6
# global attention for every 4 blocks
model.backbone.net.window_block_indexes = (
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) + list(range(12, 15)) + list(range(16, 19)) +
    list(range(20, 23)) + list(range(24, 27)) + list(range(28, 31)) + list(range(32, 35)) + list(range(36, 39))
)

optimizer.lr = 2.5e-5
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.9, num_layers=40)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None

train.max_iter = 45000
lr_multiplier.scheduler.milestones = [
    40000, train.max_iter-1,
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 500 / train.max_iter  # 2ep 118k*2/64
