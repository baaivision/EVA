from functools import partial

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import WarmupParamScheduler
from detectron2.config import LazyCall as L

from ..common.objects365_trainval_loader_lsj_1024 import dataloader
from .cascade_mask_rcnn_vitdet_b_100ep import (
    # dataloader,
    # lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = "/sharefs/baaivision/yxf/outputs/beitXclip/large-giant/150/merge30M_beit_g_patch14_224_sz224_mask105_lr1e-3_b20.98_eps1e-6_dpr0.1_ls0.0_bsz16x8x32_ep150_wmep2_cj0.0_ftpye2_ltype1_mixup0.0_abspos/checkpoint-149/mp_rank_00_model_states_renamed-s14tos16.pt"

# for o365
model.roi_heads.mask_in_features = None
model.roi_heads.mask_pooler = None
model.roi_heads.mask_head = None
model.roi_heads.num_classes = 365

# for model
model.backbone.net.img_size = 1024  # 1024
model.backbone.square_pad = 1024  # 1024
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

train.max_iter = 350057  # 25ep, (1742292+50000) * 25 / 128

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[350050, 350056],
        num_updates=train.max_iter,
    ),
    warmup_length=15000 / train.max_iter,
    warmup_factor=0.001,
)

dataloader.train.total_batch_size = 128
optimizer.lr = 1e-4
model.backbone.net.beit_like_qkv_bias = True
model.backbone.net.beit_like_gamma = False
train.output_dir = "work_dirs/o365_cascade_mask_rcnn_vitdet_1B_bs128_1024_attn_16x8"
train.checkpointer.period = 1000
train.eval_period = 5000

