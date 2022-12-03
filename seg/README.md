# EVA: Semantic Segmentation

**Table of Contents**

- [EVA: Semantic Segmentation](#eva-semantic-segmentation)
  - [Setup](#setup)
  - [Data preparation](#data-preparation)
  - [Prepare EVA pre-trained weight](#prepare-eva-pre-trained-weight)
  - [Results and Models](#results-and-models)
    - [COCO-Stuff-164K](#coco-stuff-164k)
    - [ADE20K](#ade20k)
  - [Evaluation](#evaluation)
    - [COCO-Stuff-164K](#coco-stuff-164k-1)
    - [ADE20K](#ade20k-1)
  - [Training](#training)
    - [COCO-Stuff-164K](#coco-stuff-164k-2)
    - [ADE20K](#ade20k-2)
  - [Acknowledgement](#acknowledgement)

## Setup

Install [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

```bash
# env: same as vit-adapter
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2

# compile deformable attention
cd ops & sh make.sh
```

## Data preparation

Please prepare COCO-Stuff-164K & ADE20K datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

## Prepare EVA pre-trained weight

<div align="center">

| model name | #param. |pre-training epochs on merged-30M | weight |
|------------|:------:|:------------------:|:------:|
| `eva_psz14to16` | 1.0B | 150 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt) (`2GB`) |

</div>

EVA is pre-trained with `patch_size` = `14x14`. While `eva_psz14to16` model interpolates the kernel size of `patch_embed` from `14x14` to `16x16`. This is useful for object detection, instance segmentation & semantic segmentation, *etc*. See [`interpolate_patch_14to16.py`](interpolate_patch_14to16.py) for implementation details.

## Results and Models

EVA uses ViT-Adapter + Mask2Former as the segmentation head. We evaluate EVA on COCO-Stuff-164K and ADE20K segmentation benchmarks.

We provide two sets of models:
- `EVA encoder w/o rel pos, w/o layerscale, 8 mask2former decoders`, trained on GPUs w/ 40GB VRAM.
- `EVA encoder w/  rel pos, w/  layerscale, 9 mask2former decoders`, trained on GPUs w/ 80GB VRAM.

They use slightly different hyper-parameters and yield similar results.


### COCO-Stuff-164K

<div align="center">

| init. model weight | batch size | iter | crop size | rel pos | layerscale | #dec. | mIoU (ss) | config | seg model weight |logs|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`eva_psz14to16`](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt) | 32 | 60k | 896 | ❌ | ❌ | 8 | **53.4** | [config](configs/coco_stuff164k/eva_mask2former_896_60k_cocostuff164k_ss.py) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_53p4.pth) | [training](../logs/sem_seg/ft_cocstuff164k_sem_seg_ss_53p4_training_log.txt) \| [evaluation](../logs/sem_seg/ft_cocstuff164k_sem_seg_ss_53p4.txt)
| [`eva_psz14to16`](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt) | 32 | 80k | 896 | ✔️ | ✔️ | 9 | **53.2** | [config](configs/coco_stuff164k/eva_mask2former_896_80k_cocostuff164k_ss_relpos_layerscale_9dec.py) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_relpos_layerscale_9dec_53p2.pth) | [training & evaluation](../logs/sem_seg/ft_cocstuff164k_sem_seg_ss_relpos_layerscale_9dec_53p2_training_log.txt) |


</div>

### ADE20K

<div align="center">

| init. model weight | batch size | iter | crop size | rel pos | layerscale | #dec. | mIoU | config | seg model weight |logs|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`eva_seg_coco`](https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_53p4.pth) | 64 | 20k | 896 | ❌ | ❌ | 8 | **61.5** (ss) \| **62.3** (ms) | [config](configs/ade20k/eva_mask2former_896_20k_coco164k2ade20k_ss.py) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_ade_ss61p5_ms62p3.pth) | [training](../logs/sem_seg/ft_ade20k_sem_seg_ms_62p3_training_log.txt) \| [evaluation](../logs/sem_seg/ft_ade20k_sem_seg_ms_62p3.txt)
| [`eva_seg_coco_relpos`](https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_relpos_layerscale_9dec_53p2.pth) | 32 | 40k | 896 | ✔️ | ✔️ | 9 | **61.5** (ss) \| **62.3** (ms) | [config](configs/ade20k/eva_mask2former_896_40k_coco164k2ade20k_ss_relpos_layerscale_9dec.py) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_ade_relpos_layerscale_9dec_ss61p5_ms62p3.pth) | [training & evaluation](../logs/sem_seg/ft_ade20k_sem_seg_ms_relpos_layerscale_9dec_62p3_training_log.txt) |

</div>


## Evaluation

### COCO-Stuff-164K

To evaluate EVA on **COCO-Stuff-164K** (`EVA encoder w/o rel pos, w/o layerscale, 8 mask2former decoders`) using a single node with 8 gpus:

- single-scale evaluation
```bash
SEG_CONFIG=eva_mask2former_896_60k_cocostuff164k_ss

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch \
    configs/coco_stuff164k/${SEG_CONFIG}.py \
    /path/to/eva_sem_seg_mask2former_cocostuff_53p4.pth \ # https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_53p4.pth
    --eval mIoU


# expected results
Summary:
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 74.08 | 53.36 | 66.09 |
+-------+-------+-------+
```


To evaluate EVA on **COCO-Stuff-164K** (`EVA encoder w/  rel pos, w/  layerscale, 9 mask2former decoders`) using a single node with 8 gpus:

- single-scale evaluation
```bash
SEG_CONFIG=eva_mask2former_896_80k_cocostuff164k_ss_relpos_layerscale_9dec.py

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch \
    configs/coco_stuff164k/${SEG_CONFIG}.py \
    /path/to/eva_sem_seg_mask2former_cocostuff_relpos_layerscale_9dec_53p2.pth \ # https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_relpos_layerscale_9dec_53p2.pth
    --eval mIoU


# expected results
Summary:
+-------+------+-------+
|  aAcc | mIoU |  mAcc |
+-------+------+-------+
| 73.93 | 53.2 | 66.01 |
+-------+------+-------+
```



### ADE20K

To evaluate EVA on **ADE20K** (`EVA encoder w/o rel pos, w/o layerscale, 8 mask2former decoders`) using a single node with 8 gpus:

- single-scale evaluation
```bash
SEG_CONFIG=eva_mask2former_896_20k_coco164k2ade20k_ss

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch \
    configs/ade20k/${SEG_CONFIG}.py \
    /path/to/eva_sem_seg_mask2former_ade_ss61p5_ms62p3.pth \ # https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_ade_ss61p5_ms62p3.pth
    --eval mIoU


# expected results
Summary:
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 87.15 | 61.47 | 75.75 |
+-------+-------+-------+

```


- multi-scale evaluation
```bash
SEG_CONFIG=eva_mask2former_896_20k_coco164k2ade20k_ms

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch \
    configs/ade20k/${SEG_CONFIG}.py \
    /path/to/eva_sem_seg_mask2former_ade_ss61p5_ms62p3.pth \ # https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_ade_ss61p5_ms62p3.pth
    --eval mIoU


# expected results
Summary:
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 87.35 | 62.25 | 76.14 |
+-------+-------+-------+

```



To evaluate EVA on **ADE20K** (`EVA encoder w/  rel pos, w/  layerscale, 9 mask2former decoders`) using a single node with 8 gpus:

- single-scale evaluation
```bash
SEG_CONFIG=eva_mask2former_896_40k_coco164k2ade20k_ss_relpos_layerscale_9dec

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch \
    configs/ade20k/${SEG_CONFIG}.py \
    /path/to/eva_sem_seg_mask2former_ade_relpos_layerscale_9dec_ss61p5_ms62p3.pth \ # https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_ade_relpos_layerscale_9dec_ss61p5_ms62p3.pth
    --eval mIoU


# expected results
Summary:
+------+-------+-------+
| aAcc |  mIoU |  mAcc |
+------+-------+-------+
| 87.2 | 61.54 | 76.12 |
+------+-------+-------+

```


- multi-scale evaluation
```bash
SEG_CONFIG=eva_mask2former_896_40k_coco164k2ade20k_ms_relpos_layerscale_9dec

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env test.py --launcher pytorch \
    configs/ade20k/${SEG_CONFIG}.py \
    /path/to/eva_sem_seg_mask2former_ade_relpos_layerscale_9dec_ss61p5_ms62p3.pth \ # https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_ade_relpos_layerscale_9dec_ss61p5_ms62p3.pth
    --eval mIoU


# expected results
Summary:
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 87.54 | 62.31 | 76.06 |
+-------+-------+-------+

```




## Training

### COCO-Stuff-164K

To train EVA on **COCO-Stuff-164K** (`EVA encoder w/o rel pos, w/o layerscale, 8 mask2former decoders`) using 4 nodes (`total_batch_size=32, GPU w/ 40GB VRAM`):

```bash
SEG_CONFIG=eva_mask2former_896_60k_cocostuff164k_ss
MODEL_OUTPUT_ROOT=/path/to/seg/output/
pretrained=/path/to/eva_psz14to16.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env train.py --launcher pytorch \
    configs/coco_stuff164k/${SEG_CONFIG}.py \
    --work-dir ${MODEL_OUTPUT_ROOT}/${SEG_CONFIG}/lr1.5e-5_lrd0.95_enc6_dec8 \
    --options model.pretrained=${pretrained}
```


To train EVA on **COCO-Stuff-164K** (`EVA encoder w/  rel pos, w/  layerscale, 9 mask2former decoders`) using 2 nodes (`total_batch_size=32, GPU w/ 80GB VRAM`):

```bash
SEG_CONFIG=eva_mask2former_896_80k_cocostuff164k_ss_relpos_layerscale_9dec
MODEL_OUTPUT_ROOT=/path/to/seg/output/
pretrained=/path/to/eva_psz14to16.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env train.py --launcher pytorch \
    configs/coco_stuff164k/${SEG_CONFIG}.py \
    --work-dir ${MODEL_OUTPUT_ROOT}/${SEG_CONFIG}/lr1e-5_lrd0.95_enc6_dec9 \
    --options model.pretrained=${pretrained}
```


### ADE20K

To train EVA on **ADE20K** (`EVA encoder w/o rel pos, w/o layerscale, 8 mask2former decoders`) using 8 nodes (`total_batch_size=64, GPU w/ 40GB VRAM`):

```bash
SEG_CONFIG=eva_mask2former_896_20k_coco164k2ade20k_ss
MODEL_OUTPUT_ROOT=/path/to/seg/output/
pretrained=/path/to/eva_psz14to16.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt
load_from=/path/to/eva_sem_seg_mask2former_cocostuff_53p4.pth # https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_53p4.pth

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env train.py --launcher pytorch \
    configs/coco_stuff164k/${SEG_CONFIG}.py \
    --work-dir ${MODEL_OUTPUT_ROOT}/${SEG_CONFIG}/lr2.5e-5_lrd0.95_enc6_dec8 \
    --options model.pretrained=${pretrained} \
    model.load_from=${load_from}
```


To train EVA on **ADE20K** (`EVA encoder w/  rel pos, w/  layerscale, 9 mask2former decoders`) using 2 nodes (`total_batch_size=32, GPU w/ 80GB VRAM`):

```bash
SEG_CONFIG=eva_mask2former_896_40k_coco164k2ade20k_ss_relpos_layerscale_9dec
MODEL_OUTPUT_ROOT=/path/to/seg/output/
pretrained=/path/to/eva_psz14to16.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt
load_from=/path/to/eva_sem_seg_mask2former_cocostuff_relpos_layerscale_9dec_53p2.pth # https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_relpos_layerscale_9dec_53p2.pth

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env train.py --launcher pytorch \
    configs/coco_stuff164k/${SEG_CONFIG}.py \
    --work-dir ${MODEL_OUTPUT_ROOT}/${SEG_CONFIG}/lr1e-5_lrd0.9_enc6_dec9 \
    --options model.pretrained=${pretrained} \
    model.load_from=${load_from}
```


## Acknowledgement
EVA semantic segmentation is built with [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2), [ViT-Adapter](https://arxiv.org/abs/2205.08534) and [Mask2Former](https://arxiv.org/abs/2112.01527). Thanks for their awesome work!