from detectron2.config import LazyCall as L
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation.lvis_evaluation import LVISEvaluator
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads

from ..COCO.cascade_mask_rcnn_vitdet_1B_attn_1536 import (
    dataloader,
    model,
    train,
    lr_multiplier,
    optimizer,
)

dataloader.train.dataset.names = "lvis_v1_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)
dataloader.test.dataset.names = "lvis_v1_val"
dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)

# model.roi_heads.num_classes = 1203
# model.roi_heads.box_predictor.test_score_thresh = 0.02
# model.roi_heads.box_predictor.test_topk_per_image = 300
# model.roi_heads.box_predictor.use_sigmoid_ce = True
# model.roi_heads.box_predictor.use_fed_loss = True
# model.roi_heads.box_predictor.get_fed_loss_cls_weights = lambda: get_fed_loss_cls_weights(
#     dataloader.train.dataset.names, 0.5
# )
# [model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    num_classes=1203,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            num_classes="${...num_classes}",
            test_score_thresh=0.001,
            test_topk_per_image=1000,
            cls_agnostic_bbox_reg=True,
            use_sigmoid_ce=True,
            use_fed_loss=True,
            get_fed_loss_cls_weights=lambda: get_fed_loss_cls_weights(
                dataloader.train.dataset.names, 0.5
            ),
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

# Schedule
# 100 ep = 156250 iters * 64 images/iter / 100000 images/ep
train.max_iter = 156250
train.eval_period = 5000

lr_multiplier.scheduler.milestones = [50000, 150463]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

optimizer.lr = 2e-4
