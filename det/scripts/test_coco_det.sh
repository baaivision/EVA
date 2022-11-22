OUTPUT_DIR="work_dirs/final_models/cascade_mask_rcnn_vitdet_1B_bs64_1280_attn_pt320ko365_lr3e-5_step20k"

python tools/lazyconfig_train_net.py --num-gpus 8 \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_1B_attn_1536.py \
  --eval-only \
  "train.init_checkpoint=${OUTPUT_DIR}/avg_35k_to_50k_modelonly.pth" \
  "train.output_dir=${OUTPUT_DIR}" \
  'model.roi_heads.use_soft_nms=True' \
  'model.roi_heads.method="linear"' \
  "model.roi_heads.iou_threshold=0.6" \
  'model.roi_heads.sigma=0.5' \
  'model.roi_heads.override_score_thresh=0.0' \
  "dataloader.evaluator.output_dir=${OUTPUT_DIR}"
