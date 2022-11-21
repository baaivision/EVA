#!/bin/bash

VIDEO_CONFIG=configs/kinetics600_ft.yaml
OUTPUT_ROOT=/path/to/video/output/
pretrained=pretrained/eva_video_k722.pth

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes \
--node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 \
main.py -cfg ${VIDEO_CONFIG} \
--output ${OUTPUT_ROOT} \
--accumulation-steps 4 \
--opts MODEL.PRETRAINED ${pretrained}