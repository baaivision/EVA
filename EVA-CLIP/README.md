
<div align="center">

<h2>EVA-CLIP: Improved Training Techniques for CLIP at Scale</h2>

</div>


> Mar, 21, 2023: The complete suit of EVA-CLIP (technical report, model weights, evaluation & training code) will be available in one week.

We launch EVA-CLIP, a series of models that significantly improve the efficiency and effectiveness of CLIP training. Our approach incorporates new techniques for representation learning, optimization, and augmentation, enabling EVA-CLIP to achieve superior performance compared to previous CLIP models with the same number of parameters but significantly smaller training costs.

Notably, using exclusively publicly accessible training data, our large-sized EVA-02 CLIP-L/14 can reach up to **80.4** zero-shot top-1 on ImageNet-1K, outperforming the previous largest & best open-sourced CLIP with only ~1/6 parameters and ~1/6 image-text training data. Our largest EVA-02 CLIP-E/14 with only 4 billion seen samples achieves **81.9** zero-shot top-1 accuracy on ImageNet-1K. 


**Table of Contents**

- [Summary of EVA-CLIP performance](#summary-of-eva-clip-performance)
- [Model Card](#model-card)
- [Setup](#setup)
- [Evaluation of Image Classification Performance](#evaluation-of-image-classification-performance)
  - [Evaluate EVA-CLIP on IN-1K](#evaluate-eva-clip-on-in-1k)
- [Pre-training](#pre-training)
  - [Pre-train EVA-CLIP on LAION-2B dataset](#pre-train-eva-clip-on-laion-2b-dataset)
- [Acknowledgement](#acknowledgement)


## Summary of EVA-CLIP performance

![summary_tab](assets/summary_tab.png)

## Model Card

<div align="center">

| model name | total #param. | precision | data  |  batch size |  gpus for training | IN-1K zero-shot top-1 | MSCOCO T2I R@5 |weight |
|:-----------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| `EVA01_CLIP_g_14_psz14_s11B` | 1.1B | `fp16` | [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) | 41K | 256 A100(40GB) | 78.5 | 68.5 | [🤗 HF link]() (`2.2GB`) |
| `EVA01_CLIP_g_14_plus_psz14_s11B` | 1.3B | `fp16` | Merged-2B | 114K | 112 A100(40GB) | 79.3 | 74.0 | [🤗 HF link]() (`2.6GB`) |
| `EVA02_CLIP_B_psz16_s8B` | 149M | `fp16` | Merged-2B | 131K | 64 A100(40GB) | 74.7 | 66.9 | [🤗 HF link]() (`286MB`) |
| `EVA02_CLIP_L_psz14_s4B` | 428M | `fp16` | Merged-2B | 131K | 128 A100(40GB) | 79.8 | 71.2 | [🤗 HF link]() (`817MB`) |
| `EVA02_CLIP_L_336_psz14_s6B` | 428M | `fp16` | Merged-2B | 61K | 128 A100(40GB) | 80.4 | 71.7 | [🤗 HF link]() (`817MB`) |
| `EVA02_CLIP_E_psz14_s4B.pt` | 4.7B | `fp16` | [LAION-2B](https://laion.ai/blog/laion-5b/) | 144K | 144 A100(80GB) | 81.9 | 74.7 | [🤗 HF link]() (`8.8GB`) |

</div>

To construct Merged-2B, we merged 1.6 billion samples from [LAION-2B](https://laion.ai/blog/laion-5b/) dataset with 0.4 billion samples from [COYO-700M](https://github.com/kakaobrain/coyo-dataset).

To our knowledge, EVA-CLIP is **the largest performant open-sourced CLIP model** evaluated via zero-shot classification performance, especially on mainstream classification benchmarks such as ImageNet along with its variants. 
For more details about EVA-CLIP, please refer to Section 2.3.5 of [our paper](https://arxiv.org/pdf/2211.07636.pdf).

## Setup


First, clone the repo and install required packages:
```bash
conda create --name Rei python=3.8 -y
conda activate Rei

git clone git@github.com:baaivision/EVA.git
cd EVA-CLIP
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

Then, install [Apex](https://github.com/NVIDIA/apex#linux) and [xFormer](https://github.com/facebookresearch/xformers#installing-xformers) following the official instruction. 


Core packages: 
- [Pytorch](https://pytorch.org/) version 1.12.1 
- [torchvision](https://pytorch.org/vision/stable/index.html) version 0.13.0
- [timm](https://github.com/rwightman/pytorch-image-models) version 0.5.4 
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) version 0.6.5 (`fp16` training and ZeRO optimizer)
- [Apex](https://github.com/NVIDIA/apex) (fused layer norm)
- [xFormer](https://github.com/facebookresearch/xformers) (fast and memory efficient MHSA)

## Evaluation of Image Classification Performance
### Evaluate EVA-CLIP on IN-1K
We use the standard IN-1K dataset (1.2M images). 
Download it from http://image-net.org.
Then, move and extract the training and validation images to labeled subfolders, using the [shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).

<details>
  <summary>Evaluate the <code>EVA01_CLIP_g_14_psz14_s11B</code> on <b>IN-1K val</b> using a single node with 1 gpu (click to expand).</summary>

```bash    
MODEL_NAME=EVA-ViT-g-14-X

EVAL_CKPT=/path/to/EVA01_CLIP_g_14_psz14_s11B.pt

DATA_PATH=/path/to/IN-1K/val

cd rei

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$RANK \
	--master_addr=$MASTER_ADDR --master_port=12355 --use_env training/main.py \
        --imagenet-val ${DATA_PATH} \
        --model ${MODEL_NAME} \
        --pretrained ${EVAL_CKPT} \
        --enable_deepspeed
```

</details>

<details>
  <summary>Evaluate the <code>EVA01_CLIP_g_14_plus_psz14_s11B</code> on <b>IN-1K val</b> using a single node with 1 gpu (click to expand).</summary>

```bash    
MODEL_NAME=EVA-ViT-g-14-text-H-X

EVAL_CKPT=/path/to/EVA01_CLIP_g_14_plus_psz14_s11B.pt

DATA_PATH=/path/to/IN-1K/val

cd rei

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$RANK \
	--master_addr=$MASTER_ADDR --master_port=12355 --use_env training/main.py \
        --imagenet-val ${DATA_PATH} \
        --model ${MODEL_NAME} \
        --pretrained ${EVAL_CKPT} \
        --enable_deepspeed
```

</details>

<details>
  <summary>Evaluate the <code>EVA02_CLIP_B_psz16_s8B</code> on <b>IN-1K val</b> using a single node with 1 gpu (click to expand).</summary>

```bash    
MODEL_NAME=EVA-ViT-B-16-X

EVAL_CKPT=/path/to/EVA02_CLIP_B_psz16_s8B.pt

DATA_PATH=/path/to/IN-1K/val

cd rei

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$RANK \
	--master_addr=$MASTER_ADDR --master_port=12355 --use_env training/main.py \
        --imagenet-val ${DATA_PATH} \
        --model ${MODEL_NAME} \
        --pretrained ${EVAL_CKPT} \
        --enable_deepspeed
```

</details>

<details>
  <summary>Evaluate the <code>EVA02_CLIP_L_psz14_s4B</code> on <b>IN-1K val</b> using a single node with 1 gpu (click to expand).</summary>

```bash    
MODEL_NAME=EVA-ViT-L-14-X

EVAL_CKPT=/path/to/EVA02_CLIP_L_psz14_s4B.pt

DATA_PATH=/path/to/IN-1K/val

cd rei

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$RANK \
	--master_addr=$MASTER_ADDR --master_port=12355 --use_env training/main.py \
        --imagenet-val ${DATA_PATH} \
        --model ${MODEL_NAME} \
        --pretrained ${EVAL_CKPT} \
        --enable_deepspeed
```

</details>


<details>
  <summary>Evaluate the <code>EVA02_CLIP_L_336_psz14_s6B</code> on <b>IN-1K val</b> using a single node with 1 gpu (click to expand).</summary>

```bash    
MODEL_NAME=EVA-ViT-4b-14-text-H-X

EVAL_CKPT=/path/to/EVA02_CLIP_L_336_psz14_s6B.pt

DATA_PATH=/path/to/IN-1K/val

cd rei

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$RANK \
	--master_addr=$MASTER_ADDR --master_port=12355 --use_env training/main.py \
        --imagenet-val ${DATA_PATH} \
        --model ${MODEL_NAME} \
        --pretrained ${EVAL_CKPT} \
        --enable_deepspeed
```

</details>



<details>
  <summary>Evaluate the <code>EVA02_CLIP_E_psz14_s4B</code> on <b>IN-1K val</b> using a single node with 1 gpu (click to expand).</summary>

```bash    
MODEL_NAME=EVA-ViT-L-14-X-336

EVAL_CKPT=/path/to/EVA02_CLIP_E_psz14_s4B.pt

DATA_PATH=/path/to/IN-1K/val

cd rei

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$WORLD_SIZE --node_rank=$RANK \
	--master_addr=$MASTER_ADDR --master_port=12355 --use_env training/main.py \
        --imagenet-val ${DATA_PATH} \
        --model ${MODEL_NAME} \
        --pretrained ${EVAL_CKPT} \
        --enable_deepspeed
```

</details>

## Pre-training

### Pre-train EVA-CLIP on LAION-2B dataset

We provide instruction of pre-training EVA-CLIP on LAION-2B dataset and Merged-2B dataset. 




## Acknowledgement
EVA-CLIP is built using the awesome [OpenCLIP](https://github.com/mlfoundations/open_clip), [EVA-01](https://github.com/baaivision/EVA/tree/master/EVA-01), [CLIP](https://github.com/openai/CLIP), [timm](https://github.com/rwightman/pytorch-image-models), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [Apex](https://github.com/NVIDIA/apex) and [xFormer](https://github.com/facebookresearch/xformers).

