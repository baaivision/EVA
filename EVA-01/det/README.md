# EVA: Object Detection & Instance Segmentation

**Table of Contents**

We provide fine-tuning and single-scale evaluation code on COCO & LVIS based on EVA pre-trained on Object365.
All model weights related to object detection and instance segmentation are available for the community.

- [EVA: Object Detection \& Instance Segmentation](#eva-object-detection--instance-segmentation)
  - [Setup](#setup)
  - [Data preparation](#data-preparation)
  - [Prepare Objects365 pre-trained EVA weights](#prepare-objects365-pre-trained-eva-weights)
  - [Models and results summary](#models-and-results-summary)
    - [COCO 2017 (single-scale evaluation on `val` set)](#coco-2017-single-scale-evaluation-on-val-set)
    - [LVIS v1.0 (single-scale evaluation on `val` set)](#lvis-v10-single-scale-evaluation-on-val-set)
  - [Evaluation](#evaluation)
    - [COCO 2017](#coco-2017)
    - [LVIS v1.0 `val`](#lvis-v10-val)
  - [Training](#training)
    - [COCO 2017](#coco-2017-1)
    - [LVIS v1.0](#lvis-v10)
  - [Acknowledgment](#acknowledgment)


## Setup

```bash
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html  # for soft-nms

# build EVA det / Detectron2 from source
cd /path/to/EVA/det
python -m pip install -e .
```


## Data preparation

Please prepare COCO 2017 & LVIS v1.0 datasets according to the [guidelines](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html) in Detectron2.

## Prepare Objects365 pre-trained EVA weights

<div align="center">

| model name | #param. | pre-training interations on Objects365 |                                    weight                                     |
|------------|:-------:|:--------------------------------------:|:-----------------------------------------------------------------------------:|
| `eva_o365` |  1.1B   |                  380k                  |       [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_o365.pth) (`4GB`)        |

</div>

## Models and results summary

EVA use [ViT-Det](https://arxiv.org/abs/2203.16527) + [Cascade Mask RCNN](https://arxiv.org/abs/1906.09756) as the object detection and instance segmentation head. 
We evaluate EVA on COCO 2017 and LVIS v1.0 benchmarks.


### COCO 2017 (single-scale evaluation on `val` set)

<div align="center">

| init. model weight | batch size | iter  | AP box | AP mask | config | model weight |
| :---: | :---: |:----:|:--------:|:---------------:|:---------------------------------------:|:--------------------------------------------------------------------:|
| [`eva_o365`](https://huggingface.co/BAAI/EVA/blob/main/eva_o365.pth) | 64 | 35k | **64.2** | **53.9** | [config](projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_coco_det.pth) (`4GB`)  |
| [`eva_o365`](https://huggingface.co/BAAI/EVA/blob/main/eva_o365.pth) | 64 | 45k | **63.9** | **55.0** | [config](projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_coco_seg.pth) (`4GB`)  |

</div>

### LVIS v1.0 (single-scale evaluation on `val` set)

<div align="center">

| init. model weight | batch size | iter |  AP box  |     AP mask     |                 config                  |                             model weight                             |
| :---: | :---: |:----:|:--------:|:---------------:|:---------------------------------------:|:--------------------------------------------------------------------:|
| [`eva_o365`](https://huggingface.co/BAAI/EVA/blob/main/eva_o365.pth) | 64 | 75k  | **62.2** | **55.0**| [config](projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_eva.py) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_lvis.pth) (`4GB`)  |

</div>

## Evaluation

### COCO 2017

**Object Detection**
  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=eva-exploring-the-limits-of-masked-visual)

To evaluate EVA on **COCO 2017 `val`** using a single node with 8 gpus:

```bash
python tools/lazyconfig_train_net.py --num-gpus 8 \
    --eval-only \
    --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva_1536.py \
    "train.init_checkpoint=/path/to/eva_coco_det.pth" \ # https://huggingface.co/BAAI/EVA/blob/main/eva_coco_det.pth
    "model.roi_heads.use_soft_nms=True" \
    'model.roi_heads.method="linear"' \
    "model.roi_heads.iou_threshold=0.6" \
    "model.roi_heads.override_score_thresh=0.0"
``` 

Expected results:

```bash
Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 64.164 | 81.897 | 70.561 | 49.485 | 68.088 | 77.651 |
```

**Instance Segmentation**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=eva-exploring-the-limits-of-masked-visual)

To evaluate EVA on **COCO 2017 `val`** using a single node with 8 gpus:

```bash
python tools/lazyconfig_train_net.py --num-gpus 8 \
    --eval-only \
    --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva_1536.py \
    "train.init_checkpoint=/path/to/eva_coco_seg.pth" \ # https://huggingface.co/BAAI/EVA/blob/main/eva_coco_seg.pth
    "model.roi_heads.use_soft_nms=True" \
    'model.roi_heads.method="linear"' \
    "model.roi_heads.iou_threshold=0.6" \
    "model.roi_heads.override_score_thresh=0.0" \
    "model.roi_heads.maskness_thresh=0.5" # use maskness to calibrate mask predictions
```

Expected results:

```bash
Evaluation results for segm:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 55.024 | 79.400 | 60.872 | 37.584 | 58.435 | 72.034 |
```


### LVIS v1.0 `val`

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/instance-segmentation-on-lvis-v1-0-val)](https://paperswithcode.com/sota/instance-segmentation-on-lvis-v1-0-val?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/object-detection-on-lvis-v1-0-val)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-val?p=eva-exploring-the-limits-of-masked-visual)

To evaluate EVA on **LVIS v1.0 `val`** using a single node with 8 gpus:

```bash
python tools/lazyconfig_train_net.py --num-gpus 8 \
    --eval-only \
    --config-file projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_eva_1536.py \
    "train.init_checkpoint=/path/to/eva_lvis.pth" \ # https://huggingface.co/BAAI/EVA/blob/main/eva_lvis.pth
    "dataloader.evaluator.max_dets_per_image=1000" \
    "model.roi_heads.maskness_thresh=0.5" # use maskness to calibrate mask predictions
```

Expected results

```bash
# object detection
Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  APr   |  APc   |  APf   |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 62.169 | 76.198 | 65.364 | 54.086 | 71.103 | 77.228 | 55.149 | 62.242 | 65.172 |

# instance segmentation
Evaluation results for segm:
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  APr   |  APc   |  APf   |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| 54.982 | 74.214 | 60.114 | 44.894 | 65.657 | 72.792 | 48.329 | 55.478 | 57.352 |
```



## Training

### COCO 2017

To train EVA on **COCO 2017** using 8 nodes (`total_batch_size=64`):

```bash
python tools/lazyconfig_train_net.py --num-gpus 8 \
    --num-machines $NNODES --machine-rank $NODE_RANK --dist-url "tcp://$MASTER_ADDR:60900" \
    --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py \
    "train.init_checkpoint=/path/to/eva_o365.pth" \ # https://huggingface.co/BAAI/EVA/blob/main/eva_o365.pth
    "train.output_dir=/path/to/output"
```

### LVIS v1.0

To train EVA on **LVIS v1.0** using 8 nodes (`total_batch_size=64`):

```bash
python tools/lazyconfig_train_net.py --num-gpus 8 \
    --num-machines $NNODES --machine-rank $NODE_RANK --dist-url "tcp://$MASTER_ADDR:60900" \
    --config-file projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_eva.py \
    "train.init_checkpoint=/path/to/eva_o365.pth" \ # https://huggingface.co/BAAI/EVA/blob/main/eva_o365.pth
    "train.output_dir=/path/to/output"
```

## Acknowledgment
EVA object detection and instance segmentation are built upon [Detectron2](https://github.com/facebookresearch/detectron2). Thanks for their awesome work!