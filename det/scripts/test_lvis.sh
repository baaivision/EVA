#!/bin/bash
### Platform check
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
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

ITER=74999_modelonly
OUTPUT_DIR="work_dirs/final_models/lvis_cascade_mask_rcnn_vitdet_1B_bs64_1280_attn_pt320ko365_lr2.5e-5_step70k"
# python demo1 (detectron2)
# --num-machines $RLAUNCH_REPLICA_TOTAL --machine-rank $RLAUNCH_REPLICA --dist-url "tcp://$MASTER_ADDR:60900" \
#sudo apt install net-tools -y && 
python tools/lazyconfig_train_net.py  --num-gpus 8 \
 --num-machines $RLAUNCH_REPLICA_TOTAL --machine-rank $RLAUNCH_REPLICA --dist-url "tcp://$MASTER_ADDR:60900" \
 --eval-only \
 --config-file projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_1B_attn_1536.py \
 "train.init_checkpoint=${OUTPUT_DIR}/model_00${ITER}.pth" \
 "train.output_dir=${OUTPUT_DIR}/val_${ITER}_maskness" \
 "dataloader.evaluator.max_dets_per_image=1000" \
 "dataloader.evaluator.output_dir=${OUTPUT_DIR}/val_${ITER}_maskness" \
 'model.roi_heads.maskness_thresh=0.5'
