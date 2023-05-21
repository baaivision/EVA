# EVA-02: Object Detection & Instance Segmentation

We provide evaluation and training code on Object365, COCO and LVIS datasets.
All model weights related to object detection and instance segmentation are available for the community.

**Table of Contents**

- [EVA-02: Object Detection \& Instance Segmentation](#eva-02-object-detection--instance-segmentation)
  - [EVA-02 Model Card](#eva-02-model-card)
    - [Head-to-head Comparison](#head-to-head-comparison)
      - [COCO](#coco)
      - [LVIS](#lvis)
    - [System-level Comparisons *w/o* O365 Intermediate Fine-tuning](#system-level-comparisons-wo-o365-intermediate-fine-tuning)
      - [COCO](#coco-1)
      - [LVIS](#lvis-1)
    - [Object365 Intermediate Fine-tuning](#object365-intermediate-fine-tuning)
    - [System-level Comparisons *w/* O365 Intermediate Fine-tuning](#system-level-comparisons-w-o365-intermediate-fine-tuning)
      - [COCO](#coco-2)
      - [LVIS](#lvis-2)
  - [Setup](#setup)
    - [Environment](#environment)
    - [Data](#data)
    - [EVA-02 pre-trained weight](#eva-02-pre-trained-weight)
      - [MIM Pre-trained EVA-02](#mim-pre-trained-eva-02)
      - [O365 Intermediate Fine-tuned EVA-02](#o365-intermediate-fine-tuned-eva-02)
  - [Evaluation](#evaluation)
    - [Head-to-head Comparison](#head-to-head-comparison-1)
      - [COCO](#coco-3)
      - [LVIS](#lvis-3)
    - [System-level Comparisons *w/o* O365 Intermediate Fine-tuning](#system-level-comparisons-wo-o365-intermediate-fine-tuning-1)
      - [COCO](#coco-4)
      - [LVIS](#lvis-4)
    - [System-level Comparisons *w/* O365 Intermediate Fine-tuning](#system-level-comparisons-w-o365-intermediate-fine-tuning-1)
      - [COCO](#coco-5)
      - [LVIS](#lvis-5)
  - [Training](#training)
  - [Acknowledgment](#acknowledgment)



## EVA-02 Model Card

EVA-02 uses [ViTDet](https://arxiv.org/abs/2203.16527) + [Cascade Mask RCNN](https://arxiv.org/abs/1906.09756) as the object detection and instance segmentation head. 
We mainly evaluate EVA-02 on COCO and LVIS val set.

To avoid data contamination, all LVIS models are initialized using IN-21K MIM pre-trained EVA-02. Refer to our paper for details. 


### Head-to-head Comparison

#### COCO

<div align="center">

| model name | init. ckpt | LSJ crop size | batch size | iter | AP box | AP mask | config | weight |
| --- | --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ``eva02_B_coco_bsl`` | ``eva02_B_pt_in21k_p14to16`` | ``1024x1024`` | 128 | 60k | 55.5 | 47.1 | [link](projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_B_coco_bsl.pth) |
| ``eva02_L_coco_bsl`` | ``eva02_L_pt_m38m_p14to16``  | ``1024x1024`` | 144 | 60k | 59.2 | 50.8 | [link](projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_lrd0p8.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_bsl.pth) |

</div>

#### LVIS

<div align="center">

| model name | init. ckpt | LSJ crop size | batch size | iter | AP box | AP mask | config | weight |
| --- | --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ``eva02_B_lvis_bsl`` | ``eva02_B_pt_in21k_p14to16`` | ``1024x1024`` | 128 | 50k | 47.1 | 41.4 | [link](projects/ViTDet/configs/eva2_mim_to_lvis/eva2_lvis_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_B_lvis_bsl.pth) |
| ``eva02_L_lvis_bsl`` | ``eva02_L_pt_in21k_p14to16``  | ``1024x1024`` | 128 | 40k | 55.3 | 48.6 | [link](projects/ViTDet/configs/eva2_mim_to_lvis/eva2_lvis_cascade_mask_rcnn_vitdet_l_4attn_1024_lrd0p8.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_lvis_bsl.pth) |

</div>

### System-level Comparisons *w/o* O365 Intermediate Fine-tuning

#### COCO

<div align="center">

| model name | init. ckpt | LSJ crop size | batch size | iter | AP box | AP mask | config | weight |
| --- | --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ``eva02_B_coco_sys`` | ``eva02_B_pt_in21k_p14to16`` | ``1536x1536`` | 128 | 60k | 58.9 | 50.7 | [link](projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_6attn_win32_1536_lrd0p7.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_B_coco_sys.pth) |
| ``eva02_L_coco_sys`` | ``eva02_L_pt_m38m_p14to16``  | ``1536x1536`` | 128 | 60k | 62.3 | 53.8 | [link](projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_sys.pth) |

</div>

#### LVIS

<div align="center">

| model name | init. ckpt | LSJ crop size | batch size | iter | AP box | AP mask | config | weight |
| --- | --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ``eva02_L_lvis_sys`` | ``eva02_L_pt_in21k_p14to16``  | ``1536x1536`` | 128 | 40k | 60.1 | 53.5 | [link](projects/ViTDet/configs/eva2_mim_to_lvis/eva2_lvis_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_lvis_sys.pth) |

</div>

### Object365 Intermediate Fine-tuning

<div align="center">

| model name | init. ckpt | LSJ crop size | batch size | iter | config | weight |
| --- | --- |:---:|:---:|:---:|:---:|:---:|
| ``eva02_L_m38m_to_o365`` | ``eva02_L_pt_m38m_p14to16``  | ``1536x1536`` | 160 | 400k | [link](projects/ViTDet/configs/eva2_o365/eva2_o365_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_m38m_to_o365.pth) |
| ``eva02_L_in21k_to_o365`` | ``eva02_L_pt_in21k_p14to16``  | ``1536x1536`` | 160 | 400k | [link](projects/ViTDet/configs/eva2_o365/eva2_o365_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_in21k_to_o365.pth) |

</div>


### System-level Comparisons *w/* O365 Intermediate Fine-tuning


#### COCO

<div align="center">

| model name | init. ckpt | LSJ crop size | batch size | iter | AP box | AP mask | config | weight |
| --- | --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ``eva02_L_coco_det_sys_o365`` | ``eva02_L_m38m_to_o365``  | ``1536x1536`` | 64 | 40k | **64.1** | 54.3 | [link](projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth) |
| ``eva02_L_coco_seg_sys_o365`` | ``eva02_L_m38m_to_o365``  | ``1536x1536`` | 64 | 40k | 63.9 | **55.4** | [link](projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_seg_sys_o365.pth) |

</div>

- We use different checkpoints (from the same training job) for object detection and instance segmentation tasks, since the instance segmentation part is not pre-trained on O365 and converges slower on COCO.


#### LVIS

<div align="center">

| model name | init. ckpt | LSJ crop size | batch size | iter | AP box | AP mask | config | weight |
| --- | --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ``eva02_L_lvis_sys_o365`` | ``eva02_L_in21k_to_o365``  | ``1536x1536`` | 64 | 70k | 65.2 | 57.3 | [link](projects/ViTDet/configs/eva2_o365_to_lvis/eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py) |  [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_lvis_sys_o365.pth) |

</div>



## Setup

### Environment

First, setup EVA-02 pre-training & image classification environment, and install [``mmcv==1.7.1``](https://github.com/open-mmlab/mmcv) for soft-nms.

Then, build EVA-02 det / Detectron2 from source:

```bash
cd /path/to/EVA-02/det
python -m pip install -e .
```

### Data

For Object365 (O365) dataset, download it from [here](https://open.baai.ac.cn/data-set-detail/MTI2NDc=/MTA=/true).
The file structure of O365 should look like:
```bash
o365
├── annotations
│   ├── zhiyuan_objv2_train.json
│   └── zhiyuan_objv2_val.json
├── images
│   ├── patch0
│   ├── patch1
│       ...
│   └── patch50
└── ...

```

For COCO and LVIS datasets, please follow the official [guidelines](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html) in Detectron2.

Overall, the structure of ``DETECTRON2_DATASETS`` should look like:

```bash
DETECTRON2_DATASETS
├── o365
├── coco
├── lvis
└── ...
```


### EVA-02 pre-trained weight

#### MIM Pre-trained EVA-02

<div align="center">

| model name | #params | MIM pt dataset | MIM pt epochs | weight |
|------------|:-------:|:--------------:|:-------------:|:------:|
| `eva02_B_pt_in21k_p14to16` | 86M | IN-21K | 150 | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_B_pt_in21k_p14to16.pt) |
| `eva02_L_pt_in21k_p14to16` | 304M | IN-21K | 150 | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_L_pt_in21k_p14to16.pt) |
| `eva02_L_pt_m38m_p14to16` | 304M | Merged-38M | 56 | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_L_pt_m38m_p14to16.pt) |

</div>

- `eva02_psz14to16` models interpolate the kernel size of `patch_embed` from `14x14` to `16x16`, and interpolate the `pos_embed` from `16x16` to `14x14`. This is useful for object detection, instance segmentation & semantic segmentation tasks.



#### O365 Intermediate Fine-tuned EVA-02

Please see [here](#object365-intermediate-fine-tuning).





## Evaluation


### Head-to-head Comparison

#### COCO

<details>
<summary>Evaluate <code>eva02_B_coco_bsl</code> on COCO val using a single node with 4 gpus.</summary>

```bash
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7.py \
 --eval-only \
 train.init_checkpoint=/path/to/eva02_B_coco_bsl.pth
``` 

Expected results:

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl
55.5017,74.8972,60.2801,36.3313,60.9265,72.7190
Task: segm
AP,AP50,AP75,APs,APm,APl
47.0794,71.8778,50.4881,25.9719,51.6530,67.5334
```

</details>



<details>
<summary>Evaluate <code>eva02_L_coco_bsl</code> on COCO val using a single node with 4 gpus.</summary>

```bash
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_4attn_1024_lrd0p8.py \
 --eval-only \
 train.init_checkpoint=/path/to/eva02_L_coco_bsl.pth
``` 

Expected results:

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl
59.1550,78.6420,64.0970,41.9209,64.4478,75.3628
Task: segm
AP,AP50,AP75,APs,APm,APl
50.7923,75.8581,55.2317,30.4149,55.2931,70.3713
```

</details>


#### LVIS


<details>
<summary>Evaluate <code>eva02_B_lvis_bsl</code> on LVIS val using a single node with 4 gpus.</summary>

```bash
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_mim_to_lvis/eva2_lvis_cascade_mask_rcnn_vitdet_b_4attn_1024_lrd0p7.py \
 --eval-only \
 train.init_checkpoint=/share/project/yxf/open/eva02/det/eva02_B_lvis_bsl.pth
``` 

Expected results:

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl,APr,APc,APf
47.1304,62.8418,49.3729,34.5095,58.6703,69.0790,36.3894,47.9805,50.9023
Task: segm
AP,AP50,AP75,APs,APm,APl,APr,APc,APf
41.3575,60.4043,44.2597,27.0808,54.2237,66.4821,32.2175,42.5939,43.9947
```

</details>



<details>
<summary>Evaluate <code>eva02_L_lvis_bsl</code> on LVIS val using a single node with 4 gpus.</summary>

```bash
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_mim_to_lvis/eva2_lvis_cascade_mask_rcnn_vitdet_l_4attn_1024_lrd0p8.py \
 --eval-only \
 train.init_checkpoint=/share/project/yxf/open/eva02/det/eva02_L_lvis_bsl.pth
``` 

Expected results:

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl,APr,APc,APf
55.2866,71.9707,58.1774,41.9690,67.2664,74.6449,50.6344,56.7946,55.6481
Task: segm
AP,AP50,AP75,APs,APm,APl,APr,APc,APf
48.5964,69.3752,52.5725,33.2063,61.7856,71.2531,45.5442,50.2401,48.1034
```

</details>


### System-level Comparisons *w/o* O365 Intermediate Fine-tuning

#### COCO


<details>
<summary>Evaluate <code>eva02_B_coco_sys</code> on COCO val using a single node with 4 gpus.</summary>

```bash
# evaluate object detection performance w/o maskness
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_6attn_win32_1536_lrd0p7.py \
 --eval-only \
 model.roi_heads.use_soft_nms=True \
 model.roi_heads.class_wise=True \
 model.roi_heads.method=linear \
 model.roi_heads.iou_threshold=0.6 \
 model.roi_heads.override_score_thresh=0.0 \
 train.init_checkpoint=/path/to/eva02_B_coco_sys.pth
``` 

Expected results:

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl
58.9331,77.8558,64.5855,42.1048,63.6015,74.3375
```


```bash
# evaluate instance segmentation performance w/ maskness
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_b_6attn_win32_1536_lrd0p7.py \
 --eval-only \
 model.roi_heads.use_soft_nms=True \
 model.roi_heads.class_wise=True \
 model.roi_heads.method=linear \
 model.roi_heads.iou_threshold=0.6 \
 model.roi_heads.override_score_thresh=0.0 \
 model.roi_heads.maskness_thresh=0.5 \
 train.init_checkpoint=/path/to/eva02_B_coco_sys.pth
``` 

Expected results:

```bash
Task: segm
AP,AP50,AP75,APs,APm,APl
50.6875,74.8027,55.4802,30.5407,54.2940,69.7923
```

</details>






<details>
<summary>Evaluate <code>eva02_L_coco_sys</code> on COCO val using a single node with 4 gpus.</summary>

```bash
# evaluate object detection performance w/o maskness
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py \
 --eval-only \
 model.roi_heads.use_soft_nms=True \
 model.roi_heads.class_wise=True \
 model.roi_heads.method=linear \
 model.roi_heads.iou_threshold=0.6 \
 model.roi_heads.override_score_thresh=0.0 \
 train.init_checkpoint=/path/to/eva02_L_coco_sys.pth
``` 

Expected results:

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl
62.2848,80.8003,68.0974,45.8547,66.7479,78.0017
```


```bash
# evaluate instance segmentation performance w/ maskness
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_mim_to_coco/eva2_coco_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py \
 --eval-only \
 model.roi_heads.use_soft_nms=True \
 model.roi_heads.class_wise=True \
 model.roi_heads.method=linear \
 model.roi_heads.iou_threshold=0.6 \
 model.roi_heads.override_score_thresh=0.0 \
 model.roi_heads.maskness_thresh=0.5 \
 train.init_checkpoint=/path/to/eva02_L_coco_sys.pth
``` 

Expected results:

```bash
Task: segm
AP,AP50,AP75,APs,APm,APl
53.8006,78.2082,59.0474,34.2192,57.6212,72.6909
```

</details>




#### LVIS



<details>
<summary>Evaluate <code>eva02_L_lvis_sys</code> on LVIS val using a single node with 4 gpus.</summary>

```bash
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_mim_to_lvis/eva2_lvis_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py \
 --eval-only \
 model.roi_heads.use_soft_nms=True \
 model.roi_heads.class_wise=True \
 model.roi_heads.method=linear \
 model.roi_heads.iou_threshold=0.6 \
 model.roi_heads.maskness_thresh=0.5 \
 train.init_checkpoint=/path/to/eva02_L_lvis_sys.pth
``` 

Expected results:

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl,APr,APc,APf
60.0506,74.5881,63.7702,48.6573,70.7069,77.6001,53.1295,61.7127,61.2374
Task: segm
AP,AP50,AP75,APs,APm,APl,APr,APc,APf
53.5098,72.7459,57.9503,40.0757,65.5123,73.5845,47.5871,55.2739,54.1442
```

</details>



### System-level Comparisons *w/* O365 Intermediate Fine-tuning


#### COCO



<details>
<summary>Evaluate <code>eva02_L_coco_det_sys_o365</code> on COCO val using a single node with 4 gpus.</summary>

```bash
# evaluate object detection performance w/o maskness
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py \
 --eval-only \
 train.model_ema.use_ema_weights_for_eval_only=True \
 model.roi_heads.use_soft_nms=True \
 model.roi_heads.class_wise=True \
 model.roi_heads.method=linear \
 model.roi_heads.iou_threshold=0.6 \
 model.roi_heads.override_score_thresh=0.0 \
 train.init_checkpoint=/path/to/eva02_L_coco_det_sys_o365.pth
``` 

Expected results:

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl
64.1442,82.1375,70.2666,48.8727,68.5649,78.1375
```

</details>



<details>
<summary>Evaluate <code>eva02_L_coco_seg_sys_o365</code> on COCO val using a single node with 4 gpus.</summary>

```bash
# evaluate instance segmentation performance w/ maskness
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py \
 --eval-only \
 train.model_ema.use_ema_weights_for_eval_only=True \
 model.roi_heads.use_soft_nms=True \
 model.roi_heads.class_wise=True \
 model.roi_heads.method=linear \
 model.roi_heads.iou_threshold=0.6 \
 model.roi_heads.override_score_thresh=0.0 \
 model.roi_heads.maskness_thresh=0.5 \
 train.init_checkpoint=/path/to/eva02_L_coco_seg_sys_o365.pth
``` 

Expected results:

```bash
Task: segm
AP,AP50,AP75,APs,APm,APl
55.3584,79.7107,61.4651,36.4061,59.0486,73.1774
```

</details>



#### LVIS



<details>
<summary>Evaluate <code>eva02_L_lvis_sys_o365</code> on LVIS val using a single node with 4 gpus.</summary>

```bash
python tools/lazyconfig_train_net.py \
 --num-gpus 4  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
 --config-file projects/ViTDet/configs/eva2_o365_to_lvis/eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py \
 --eval-only \
 train.model_ema.use_ema_weights_for_eval_only=True \
 model.roi_heads.use_soft_nms=True \
 model.roi_heads.class_wise=True \
 model.roi_heads.method=linear \
 model.roi_heads.iou_threshold=0.6 \
 model.roi_heads.maskness_thresh=0.5 \
 dataloader.evaluator.max_dets_per_image=1000 \
 train.init_checkpoint=/path/to/eva02_L_lvis_sys_o365.pth
``` 

Expected results:

```bash
Task: bbox
AP,AP50,AP75,APs,APm,APl,APr,APc,APf
65.2244,78.9298,68.9737,55.3842,75.1310,79.8346,59.7070,66.8476,65.8378
Task: segm
AP,AP50,AP75,APs,APm,APl,APr,APc,APf
57.3209,76.9294,62.1691,44.9472,68.7592,74.9193,52.4748,58.9491,57.6336
```

</details>








## Training

Please select ``config.py`` and the corresponding ``init_checkpoint.pth`` based on [Model Card]((#eva-02-model-card)).

All configs can be trained with:

```bash
python tools/lazyconfig_train_net.py \
    --num-gpus 8  --num-machines ${WORLD_SIZE} --machine-rank ${RANK} --dist-url "tcp://$MASTER_ADDR:60900" \
    --config-file /path/to/config.py \
    train.init_checkpoint=/path/to/init_checkpoint.pth \
    train.output_dir=/path/to/output
```





## Acknowledgment
EVA-02 object detection and instance segmentation are built upon [Detectron2](https://github.com/facebookresearch/detectron2).
