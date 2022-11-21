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



# SEG_CONFIG=upernet_beitXclip_adapter_giant_768_relpos_80k_ade20k_ss
# SEG_CONFIG=upernet_beitXclip_adapter_giant_768_relpos_80k_ade20k_ms
# SEG_CONFIG=mask2former_beitXclip_adapter_giant_896_20k_coco164k2ade20k_ss
SEG_CONFIG=mask2former_beitXclip_adapter_giant_896_20k_coco164k2ade20k_ms



# seg train
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$RLAUNCH_REPLICA_TOTAL --node_rank=$RLAUNCH_REPLICA \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env /sharefs/baaivision/yxf/projects/ViT-Adapter/segmentation/test.py --launcher pytorch \
    /sharefs/baaivision/yxf/projects/ViT-Adapter/segmentation/configs/ade20k/${SEG_CONFIG}.py \
    /sharefs/baaivision/yxf/projects/ViT-Adapter/segmentation/eva_sem_seg_mask2former_ade_ss61p5_ms62p3.pth \
    --eval mIoU


# +-------+-------+-------+
# |  aAcc |  mIoU |  mAcc |
# +-------+-------+-------+
# | 87.15 | 61.47 | 75.75 |
# +-------+-------+-------+


# +-------+-------+-------+
# |  aAcc |  mIoU |  mAcc |
# +-------+-------+-------+
# | 87.35 | 62.25 | 76.14 |
# +-------+-------+-------+