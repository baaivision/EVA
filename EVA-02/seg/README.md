# EVA-02: Semantic Segmentation on ADE20K using UperNet

**Table of Contents**

- [EVA-02: Semantic Segmentation on ADE20K using UperNet](#eva-02-semantic-segmentation-on-ade20k-using-upernet)
  - [EVA-02 Model Card](#eva-02-model-card)
  - [Setup](#setup)
    - [Environment](#environment)
    - [Data](#data)
    - [EVA-02 pre-trained weight](#eva-02-pre-trained-weight)
  - [Evaluation](#evaluation)
  - [Training](#training)
  - [Acknowledgement](#acknowledgement)



## EVA-02 Model Card

<div align="center">

| model name | init. ckpt | batch size | iter | crop size | seg head dim | mIoU (ss) | mIoU (ms) | config | logs | weight |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| ``eva02_B_ade_seg_upernet_sz512`` | ``eva02_B_pt_in21k_p14to16`` | 32 | 60k | ``512x512`` | 768 | 55.3 | - | [link](configs/eva02/upernet/upernet_eva02_base_12_512_slide_60k.py) | [link](https://gist.github.com/Yuxin-CV/d04f73b300c798ba3882e9f6c04be224#file-eva02_b_ade_seg_upernet_sz512) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/seg/eva02_B_ade_seg_upernet_sz512.pth) |
| ``eva02_L_ade_seg_upernet_sz512`` | ``eva02_L_pt_m38m_p14to16``  | 16 | 80k | ``512x512`` | 1024 | 59.8 | 60.4 | [link](configs/eva02/upernet/upernet_eva02_large_24_512_slide_80k.py) | [link](https://gist.github.com/Yuxin-CV/d04f73b300c798ba3882e9f6c04be224#file-eva02_l_ade_seg_upernet_sz512) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/seg/eva02_L_ade_seg_upernet_sz512.pth) |
| ``eva02_L_ade_seg_upernet_sz640`` | ``eva02_L_pt_m38m_p14to16``  | 16 | 80k | ``640x640`` | 1536 | 60.1 | - | [link](configs/eva02/upernet/upernetpro_eva02_large_24_640_slide_80k.py) | [link](https://gist.github.com/Yuxin-CV/d04f73b300c798ba3882e9f6c04be224#file-eva02_l_ade_seg_upernet_sz640) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/seg/eva02_L_ade_seg_upernet_sz640.pth) |
</div>





## Setup

### Environment

First, setup EVA-02 pre-training & image classification environment.

Then, install [``mmcv==1.7.1``](https://github.com/open-mmlab/mmcv) and [``mmsegmentation==0.29.1``](https://github.com/open-mmlab/mmsegmentation).


### Data 

Please prepare ADE20K datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k) in MMSegmentation.



### EVA-02 pre-trained weight

<div align="center">

| model name | #params | MIM pt dataset | MIM pt epochs | weight |
|------------|:-------:|:--------------:|:-------------:|:------:|
| `eva02_B_pt_in21k_p14to16` | 86M | IN-21K | 150 | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_B_pt_in21k_p14to16.pt) |
| `eva02_L_pt_m38m_p14to16` | 304M | Merged-38M | 56 | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_L_pt_m38m_p14to16.pt) |

</div>

- `eva02_psz14to16` models interpolate the kernel size of `patch_embed` from `14x14` to `16x16`, and interpolate the `pos_embed` from `16x16` to `14x14`. This is useful for object detection, instance segmentation & semantic segmentation tasks.





## Evaluation


<details>

<summary>Evaluate <code>eva02_B_ade_seg_upernet_sz512</code> on ADE20K using a single node with 4 gpus.</summary>


```bash
SEG_CONFIG=configs/eva02/upernet/upernet_eva02_base_12_512_slide_60k.py
EVAL_CKPT=/path/to/eva02_B_ade_seg_upernet_sz512.pth

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=11235 \
--use_env test.py --launcher pytorch \
    ${SEG_CONFIG} \
    ${EVAL_CKPT} \
    --eval mIoU


# expected results
Summary:
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 85.41 | 55.25 | 66.81 |
+-------+-------+-------+

```

</details>





<details>
<summary>Evaluate <code>eva02_L_ade_seg_upernet_sz512</code> on ADE20K using a single node with 4 gpus.</summary>

- single-scale evaluation 
```bash
SEG_CONFIG=configs/eva02/upernet/upernet_eva02_large_24_512_slide_80k.py
EVAL_CKPT=/path/to/eva02_L_ade_seg_upernet_sz512.pth

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=11235 \
--use_env test.py --launcher pytorch \
    ${SEG_CONFIG} \
    ${EVAL_CKPT} \
    --eval mIoU


# expected results
Summary:
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 86.67 | 59.77 | 72.05 |
+-------+-------+-------+

```

- multi-scale evaluation 
```bash
SEG_CONFIG=configs/eva02/upernet/upernet_eva02_large_24_512_slide_ms_eval.py
EVAL_CKPT=/path/to/eva02_L_ade_seg_upernet_sz512.pth

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=11235 \
--use_env test.py --launcher pytorch \
    ${SEG_CONFIG} \
    ${EVAL_CKPT} \
    --eval mIoU


# expected results
Summary:                                                                                                                                                                             
+-------+-------+-------+                
|  aAcc |  mIoU |  mAcc |                                                                                                                 
+-------+-------+-------+       
| 86.98 | 60.42 | 72.54 |                                                                                                                 
+-------+-------+-------+  

```

</details>






<details>

<summary>Evaluate <code>eva02_L_ade_seg_upernet_sz640</code> on ADE20K using a single node with 4 gpus.</summary>


```bash
SEG_CONFIG=configs/eva02/upernet/upernetpro_eva02_large_24_640_slide_80k.py
EVAL_CKPT=/path/to/eva02_L_ade_seg_upernet_sz640.pth

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=11235 \
--use_env test.py --launcher pytorch \
    ${SEG_CONFIG} \
    ${EVAL_CKPT} \
    --eval mIoU


# expected results
Summary:
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 86.83 | 60.05 | 72.17 |
+-------+-------+-------+

```

</details>






## Training



<details>

<summary>Train <code>eva02_B_ade_seg_upernet_sz512</code> on ADE20K using a single node with 4 gpus.</summary>


```bash
SEG_CONFIG=configs/eva02/upernet/upernet_eva02_base_12_512_slide_60k.py
PRETRAIN_CKPT=/path/to/eva02_B_pt_in21k_p14to16.pt

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=11235 \
--use_env train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT}

```

</details>





<details>

<summary>Train <code>eva02_L_ade_seg_upernet_sz512</code> on ADE20K using 2 nodes with 8 gpus.</summary>


```bash
SEG_CONFIG=configs/eva02/upernet/upernet_eva02_large_24_512_slide_80k.py
PRETRAIN_CKPT=/path/to/eva02_L_pt_m38m_p14to16.pt

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=11235 \
--use_env train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT}

```

</details>






<details>

<summary>Train <code>eva02_L_ade_seg_upernet_sz640</code> on ADE20K using 2 nodes with 8 gpus.</summary>


```bash
SEG_CONFIG=configs/eva02/upernet/upernetpro_eva02_large_24_640_slide_80k.py
PRETRAIN_CKPT=/path/to/eva02_L_pt_m38m_p14to16.pt

python -m torch.distributed.launch --nproc_per_node8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=11235 \
--use_env train.py --launcher pytorch \
    ${SEG_CONFIG} \
    --seed 0 --deterministic \
    --options model.backbone.pretrained=${PRETRAIN_CKPT}

```

</details>



## Acknowledgement
EVA-02 semantic segmentation is built with [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit/semantic_segmentation).