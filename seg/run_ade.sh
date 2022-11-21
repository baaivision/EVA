#!/bin/bash
### Platform check

export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export PATH=/usr/local/nvidia/bin:$PATH

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_DEBUG=info
export OMP_NUM_THREADS=4
# ulimit -l 131072
export JOB_NAME=$(cat /etc/hostname | cut -d '-' -f 1,2,3)
export MASTER_FILE=$HOME/master_ip.${JOB_NAME}

if [ -z "$RLAUNCH_REPLICA_TOTAL" ]; then
        export RLAUNCH_REPLICA_TOTAL=1
fi

if [ -z "$RLAUNCH_REPLICA" ]; then
        export RLAUNCH_REPLICA=0
fi

if [ "$RLAUNCH_REPLICA" == "0" ]; then
        ifconfig $NCCL_SOCKET_IFNAME | grep inet | grep -v inet6 | awk '{print $2}' > ${MASTER_FILE}
fi

function finish {
        rm -rf ${MASTER_FILE}
}

trap finish EXIT INT TERM

while [ ! -f ${MASTER_FILE} ]; do
        echo "wait ${MASTER_FILE}..."
        ls > /dev/null && sleep 1;
done

export MASTER_ADDR=$(cat ${MASTER_FILE})
echo "master_ip: $MASTER_ADDR"
DIST_URL="tcp://$MASTER_ADDR:60900"




SEG_CONFIG=mask2former_beitXclip_adapter_giant_896_80k_coco164k2ade20k_ss

MODEL_OUTPUT_ROOT=/sharefs/baaivision/yxf/outputs/beitXclip/large-giant/150/merge30M_beit_g_patch14_224_sz224_mask105_lr1e-3_b20.98_eps1e-6_dpr0.1_ls0.0_bsz16x8x32_ep150_wmep2_cj0.0_ftpye2_ltype1_mixup0.0_abspos/seg


# # seg train
# /sharefs/baaivision/yxf/projects/ViT-Adapter/segmentation/dist_train.sh \
#     /sharefs/baaivision/yxf/projects/ViT-Adapter/segmentation/configs/ade20k/${SEG_CONFIG}.py 8 \
#     --work-dir ${MODEL_OUTPUT_ROOT}/test/${SEG_CONFIG} \
#     --auto-resume 


# seg train
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$RLAUNCH_REPLICA_TOTAL --node_rank=$RLAUNCH_REPLICA \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env /sharefs/baaivision/yxf/projects/ViT-Adapter/segmentation/train.py --launcher pytorch \
    /sharefs/baaivision/yxf/projects/ViT-Adapter/segmentation/configs/ade20k/${SEG_CONFIG}.py \
    --work-dir ${MODEL_OUTPUT_ROOT}/${SEG_CONFIG}/lr1e-5_lrd0.95_deterministic \
    --no-validate \
    --seed 0 \
    --deterministic
#     --resume-from /sharefs/baaivision/yxf/outputs/beitXclip/large-giant/150/merge30M_beit_g_patch14_224_sz224_mask105_lr1e-3_b20.98_eps1e-6_dpr0.1_ls0.0_bsz16x8x32_ep150_wmep2_cj0.0_ftpye2_ltype1_mixup0.0_abspos/seg/mask2former_beitXclip_adapter_giant_896_20k_coco164k2ade20k_ss/lr2.5e-5_lrd0.95/latest.pth