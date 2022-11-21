#!/bin/bash

VIDEO_CONFIG=configs/kinetics722_intermediate_ft.yaml
OUTPUT_ROOT=/path/to/video/output/
pretrained=/path/to/eva_psz14.pt
    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes \
--node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 \
main.py -cfg ${VIDEO_CONFIG} \
--output ${OUTPUT_ROOT} \
--accumulation-steps 1 \
--opts MODEL.PRETRAINED ${pretrained}