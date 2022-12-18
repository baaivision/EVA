# EVA: Pre-training and Image Classification 

**Table of Contents**

- [EVA: Pre-training and Image Classification](#eva-pre-training-and-image-classification)
  - [EVA Model Card](#eva-model-card)
  - [Performance of **MIM** pre-trained EVA encoder on ImageNet-1K](#performance-of-mim-pre-trained-eva-encoder-on-imagenet-1k)
  - [Performance of **EVA-CLIP** vision encoder on ImageNet-1K](#performance-of-eva-clip-vision-encoder-on-imagenet-1k)
  - [**EVA-L**: Learning better MIM representations from EVA-CLIP](#eva-l-learning-better-mim-representations-from-eva-clip)
    - [Performance of **EVA-L** on ImageNet-1K](#performance-of-eva-l-on-imagenet-1k)
    - [Comparisons with other large-sized models on ImageNet-1K](#comparisons-with-other-large-sized-models-on-imagenet-1k)
  - [Setup](#setup)
  - [Evaluate **EVA** on ImageNet-1K](#evaluate-eva-on-imagenet-1k)
  - [Evaluate **EVA** on ImageNet-1K variants (IN-V2, IN-ReaL, IN-Adv., IN-Ren., IN-Ske., ObjectNet)](#evaluate-eva-on-imagenet-1k-variants-in-v2-in-real-in-adv-in-ren-in-ske-objectnet)
  - [Evaluate **EVA-CLIP** on ImageNet-1K](#evaluate-eva-clip-on-imagenet-1k)
    - [linear probing](#linear-probing)
    - [fine-tuning](#fine-tuning)
  - [Pre-train **EVA** on the merged-30M image dataset](#pre-train-eva-on-the-merged-30m-image-dataset)
  - [Intermediate Fine-tune MIM pre-trained **EVA** on ImageNet-21K](#intermediate-fine-tune-mim-pre-trained-eva-on-imagenet-21k)
  - [Fine-tuning **EVA** on ImageNet-1K with ImageNet-21K intermediate fine-tuned checkpoint](#fine-tuning-eva-on-imagenet-1k-with-imagenet-21k-intermediate-fine-tuned-checkpoint)
  - [Transferring **EVA-CLIP** vision encoder to ImageNet-1K](#transferring-eva-clip-vision-encoder-to-imagenet-1k)
    - [linear probing](#linear-probing-1)
    - [fine-tuning](#fine-tuning-1)
  - [Acknowledgement](#acknowledgement)

## EVA Model Card

We provide **all pre-trained & fine-tuned** EVAs for the community. 
The following table summarizes the basic statistics of MIM pre-trained EVA and image classification EVA.

<div align="center">

| model name | #param. | MIM pt ep | IN-21K ft ep | IN-1K ft ep | IN-1K top-1 |weight |
|------------|:------:|:------------------:|:------:|:------:|:------:|:------:|
| `eva_psz14` | 1.0B | 150 | - | - | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14.pt) (`2GB`) |
| `eva_psz14to16` | 1.0B | 150 | - | - | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt) (`2GB`) | 
| `eva_21k_224px_psz14` | 1.0B | 150 | 60 | - | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_21k_224px_psz14.pt) (`2GB`) |
| `eva_21k_1k_336px_psz14_ema` | 1.0B | 150 | 60 | 10 | 89.6 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt) (`4GB`) |
| `eva_21k_1k_560px_psz14_ema` | 1.0B | 150 | 60 | 15 | 89.7 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_560px_psz14_ema_89p7.pt) (`4GB`) |

</div>

- `eva_psz14to16` model interpolates the kernel size of `patch_embed` from `14x14` to `16x16`. This is useful for object detection, instance segmentation & semantic segmentation, *etc*. See [`interpolate_patch_14to16.py`](interpolate_patch_14to16.py) for implementation details.
- For MIM pre-trained EVA and EVA-CLIP, we use `deepspeed` `fp16` format. IN-1K fine-tuned EVA weights are larger (`4GB` *v.s.* `2GB`) because ema updates models with `fp32` format. The weights of other downstream tasks are also with `fp32` format. 


<!-- ## Summary of EVA's image classification performance -->

## Performance of **MIM** pre-trained EVA encoder on ImageNet-1K

<div align="center">

| model | [IN-1K](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet.csv) | [IN-V2](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv) | [IN-ReaL](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-real.csv) | [IN-Adv.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-a.csv) | [IN-Ren.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv) | [IN-Ske.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv) | ObjectNet |
|:------------:|:------------------:|:------:|:------:| :------:|:------:|:------:|:------:|
| EVA (`336px`) | 89.6 | 81.6 | 90.8 | 86.2 | 88.3 | 67.7 | 60.9 |

</div>

For reference, [timm](https://github.com/rwightman/pytorch-image-models) collects some open-sourced state-of-the-art models' image classification results [here](https://github.com/rwightman/pytorch-image-models/tree/main/results) ([IN-1K](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet.csv), [IN-V2](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv), [IN-ReaL](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-real.csv), [IN-Adv.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-a.csv), [IN-Ren.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv), [IN-Ske.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv)).

Compared with other open-sourced models, EVA achieves **state-of-the-art** performance in all the classification benchmarks. 

## Performance of [**EVA-CLIP**](../clip/README.md) vision encoder on ImageNet-1K

<div align="center">

\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/self-supervised-image-classification-with-2)](https://paperswithcode.com/sota/self-supervised-image-classification-with-2?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/self-supervised-image-classification-with-3)](https://paperswithcode.com/sota/self-supervised-image-classification-with-3?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/self-supervised-image-classification-with)](https://paperswithcode.com/sota/self-supervised-image-classification-with?p=eva-exploring-the-limits-of-masked-visual) 

| model | zero-shot (`224px`) | linear probing (`224px`) | linear probing (`336px`) | fine-tuning (`224px`) | fine-tuning (`336px`) |
|:-----:|:------:|:------:|:------:|:------:|:------:| 
| [EVA-CLIP](../clip/README.md) | **78.5** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) \| [log](https://wandb.ai/baaivision/eva-clip/reports/ViT-g-14--VmlldzoyOTkwMDYy)) | **86.5** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz224_lincls_86p5.pth) \| [log](../logs/cls/linear_eva_clip_vision_enc_1k_cls_sz224_86p5.txt)） | **86.5** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz336_lincls_86p5.pth) \| [log](../logs/cls/linear_eva_clip_vision_enc_1k_cls_sz336_86p5.txt)) | **89.1** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz224_ftcls_89p1.pt) \| [log](../logs/cls/ft_eva_clip_vision_enc_1k_cls_sz224_89p1.txt)) | **89.4** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz336_ftcls_89p4.pt) \| [log](../logs/cls/ft_eva_clip_vision_enc_1k_cls_sz336_89p4.txt)) |

</div>

We also evaluate the transfer learning ability of [EVA-CLIP]((../clip/README.md)), which achieves the **state-of-the-art** top-1 accuracy on ImageNet-1K among all self-supervised learning approaches.


## **EVA-L**: Learning better MIM representations from EVA-CLIP

We show EVA-CLIP is not only performant in [zero-shot recognition](../clip/README.md), but also can improve the representation quality of MIM pre-training.

EVA-L is a vanilla ViT-Large encoder (`#layer=24; dim=1024; patch_size=14x14; #param: 303M`) pre-trained via MIM with vision features from EVA-CLIP as prediction targets. Therefore, during pre-training EVA-L learns MIM pre-text task while distills knowledge from a stronger teacher.  

We adopt the MAE-style MIM pre-training with an asymmetric
encoder-decoder architecture ([`modeling_mae_pretrain.py`](./modeling_mae_pretrain.py)), and we provide the MIM-only pre-trained checkpoint (`dataset / schedule: IN-21K / 150 epochs`) as well as MIM pre-trained + supervised intermediate fine-tuned checkpoint (`dataset / schedule: IN-21K / 90 epochs`) for the community.


<div align="center">

| model name | enc #param. | IN-21K pt ep | IN-21K ft ep | weight | pt log |
|:-----|:------:|:------:|:------:|:------:| :------: | 
| `eva_l_psz14` | 303M | 150 | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_l_psz14.pt) | [link](../logs/pt/eva-l_mim_pt_eva-clip-target_21k_150ep_log.txt) |
| `eva_l_psz14_21k_ft` | 303M | 150 | 90  | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_l_psz14_21k_ft.pt) | [link](../logs/cls/eva-l_intermed_ft_21k_90ep_sz224.txt) |

</div>

> Notice that for MAE-style ViTs, `q,k,v` all have bias term, which is different from the BEiT-style ViTs that only `q&v` have bias.  

### Performance of **EVA-L** on ImageNet-1K

<div align="center">

| model | init. ckpt | resolution | #param. | top-1 | weight | ft log |
|:-----:|:------:|:------:|:------:|:------:|:------:| :------: | 
| EVA-L | `eva_l_psz14` | `196x196` | 304M | 88.0 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_l_psz14_196px_1k_ft_88p0.pt) | [link](../logs/cls/eva-l_ft_1k_cls_sz196_50ep_88p0.txt) |
| EVA-L | `eva_l_psz14` | `336x336` | 304M | 88.6 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_l_psz14_336px_1k_ft_88p65.pt) | [link](../logs/cls/eva-l_ft_1k_cls_sz336_50ep_88p65.txt) |
| EVA-L | `eva_l_psz14_21k_ft` | `196x196` | 304M | 88.6 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_l_psz14_196px_21k_to_1k_ft_88p6.pt) | [link](../logs/cls/eva-l_ft_21k_to_1k_cls_sz196_20ep_88p6.txt) |
| EVA-L | `eva_l_psz14_21k_ft` | `336x336` | 304M | **89.2** | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_l_psz14_336px_21k_to_1k_ft_89p2.pt) | [link](../logs/cls/eva-l_ft_21k_to_1k_cls_sz336_20ep_89p2.txt) |

</div>


### Comparisons with other large-sized models on ImageNet-1K

<div align="center">

| model | resolution | #param. | top-1 | 
|:-----|:------:|:------:|:------:|
| [InternImage-XL](https://github.com/OpenGVLab/InternImage#main-results-on-imagenet-with-pretrained-models) | `384x384` | 335M | 88.0 | 
| [BEiT-L/16](https://github.com/microsoft/unilm/tree/master/beit#fine-tuning-on-imagenet-1k-image-classification) | `512x512` | 306M | 88.6 |
| [BEiTv2-L/16](https://github.com/microsoft/unilm/blob/master/beit2/get_started_for_image_classification.md#model-zoo) (prev. best) | `384x384` | 304M | 89.0 |
| **EVA-L/14** | `336x336` | 304M | **89.2** | 

</div>

**EVA-L** can reach up to **89.2** top-1 accuracy on ImageNet-1K, which is very similar to the fine-tuned EVA-CLIP teacher (89.4 top-1 accuracy). To our knowledge, EVA-L is the best open-sourced large-sized vision encoder to date.


## Setup


First, clone the repo and install required packages:
```bash
conda create --name eva python=3.8 -y
conda activate eva

git clone git@github.com:baaivision/EVA.git
cd eva
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

The core packages including: [Pytorch](https://pytorch.org/) version 1.12.0, [torchvision](https://pytorch.org/vision/stable/index.html) version 0.13.0, [timm](https://github.com/rwightman/pytorch-image-models) version 0.5.4 and [DeepSpeed](https://github.com/microsoft/DeepSpeed) version 0.7.5 *etc*.




## Evaluate **EVA** on ImageNet-1K

We use the standard ImageNet-1K dataset. 
Download it from http://image-net.org.
Then, move and extract the training and validation images to labeled subfolders, using the [shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).



<details>
  <summary>Evaluate the fine-tuned EVA (<code>336px, patch_size=14</code>) on <b>ImageNet-1K val</b> with a single node (click to expand).</summary>

```bash    
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_21k_1k_336px_psz14_ema_89p6.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt

DATA_PATH=/path/to/ImageNet-1K/


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 89.622 Acc@5 98.930 loss 0.948
```

</details>

<details>
 <summary>Evaluate the fine-tuned EVA (<code>560px, patch_size=14</code>) on <b>ImageNet-1K val</b> with a single node (click to expand).</summary>
 
```bash     
MODEL_NAME=eva_g_patch14

sz=560
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_21k_1k_560px_psz14_ema_89p7.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_560px_psz14_ema_89p7.pt

DATA_PATH=/path/to/ImageNet-1K/


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 89.712 Acc@5 98.958 loss 0.881
```

</details>


## Evaluate **EVA** on ImageNet-1K variants (IN-V2, IN-ReaL, IN-Adv., IN-Ren., IN-Ske., ObjectNet)

<details>
 <summary>Evaluate the fine-tuned EVA (<code>336px, patch_size=14</code>) on <b>ImageNet-V2</b> with a single node (click to expand).</summary>

```bash     
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_21k_1k_336px_psz14_ema_89p6.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt

DATA_PATH=/path/to/imagenetv2/ImageNetV2-matched-frequency


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --robust_test 'imagenet_v2' \
        --data_path ${DATA_PATH} \
        --eval_data_path ${DATA_PATH} \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 81.570 Acc@5 96.230 loss 1.274
```

</details>


<details>
<summary>Evaluate the fine-tuned EVA (<code>336px, patch_size=14</code>) on <b>ImageNet-ReaL</b> with a single GPU on a single node (click to expand).</summary>

```bash     
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_21k_1k_336px_psz14_ema_89p6.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt

DATA_PATH=/path/to/ImageNet-1K


python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --real_labels real.json \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 90.828 Acc@5 98.683 loss 0.947
```

</details>


<details>

<summary>Evaluate the fine-tuned EVA (<code>336px, patch_size=14</code>) on <b>ImageNet-Adversarial</b> with a single node (click to expand).</summary>

```bash     
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_21k_1k_336px_psz14_ema_89p6.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt

DATA_PATH=/path/to/imagenet-a

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --robust_test 'imagenet_a' \
        --data_path ${DATA_PATH} \
        --eval_data_path ${DATA_PATH} \
        --nb_classes 200 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 86.154 Acc@5 96.509 loss 0.979
```

</details>


<details>
<summary>Evaluate the fine-tuned EVA (<code>336px, patch_size=14</code>) on <b>ImageNet-Rendition</b> with a single node (click to expand).</summary>

```bash     
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
crop_pct=1.0


EVAL_CKPT=/path/to/eva_21k_1k_336px_psz14_ema_89p6.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt

DATA_PATH=/path/to/imagenet-r

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --robust_test 'imagenet_r' \
        --data_path ${DATA_PATH} \
        --eval_data_path ${DATA_PATH} \
        --nb_classes 200 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 88.283 Acc@5 95.830 loss 0.965
```

</details>


<details>
<summary>Evaluate the fine-tuned EVA (<code>336px, patch_size=14</code>) on <b>ImageNet-Sketch</b> with a single node (click to expand).</summary>

```bash     
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_21k_1k_336px_psz14_ema_89p6.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt

DATA_PATH=/path/to/imagenet_sketch


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH} \
        --eval_data_path ${DATA_PATH} \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 67.724 Acc@5 87.964 loss 1.955
```

</details>


<details>
<summary>Evaluate the fine-tuned EVA (<code>336px, patch_size=14</code>) on <b>ObjectNet</b> with a single node (click to expand).</summary>

```bash     
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_21k_1k_336px_psz14_ema_89p6.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt

DUMMY_DATA_PATH=/path/to/ImageNet-1K
DATA_PATH=/sharefs/baai-mmdataset/clip_benchmark_datasets/objectnet/objectnet-1.0/images

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --robust_test 'objectnet' \
        --data_path ${DUMMY_DATA_PATH}/train \
        --eval_data_path ${DATA_PATH} \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 60.907 Acc@5 82.768 loss 2.305
```

</details>


## Evaluate **EVA-CLIP** on ImageNet-1K


### linear probing 


<details>
<summary>Evaluate the linear probing performance of EVA-CLIP vision encoder (<code>224px, patch_size=14</code>) on <b>ImageNet-1K val</b> with a single node (click to expand).</summary>

```bash    
MODEL_NAME=eva_g_patch14

sz=224
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_clip_vis_enc_sz224_lincls_86p5.pth # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz224_lincls_86p5.pth

DATA_PATH=/path/to/ImageNet-1K/


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --linear_probe \
        --use_cls \
        --dist_eval \
        --eval
```

Expected results:
```
* Acc@1 86.462 Acc@5 98.034 loss 0.479
```

</details>


<details>
<summary>Evaluate the linear probing performance of EVA-CLIP vision encoder (<code>336px, patch_size=14</code>) on <b>ImageNet-1K val</b> with a single node (click to expand).</summary>

```bash    
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_clip_vis_enc_sz336_lincls_86p5.pth # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz336_lincls_86p5.pth

DATA_PATH=/path/to/ImageNet-1K/


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --linear_probe \
        --use_cls \
        --dist_eval \
        --eval
```

Expected results:
```
* Acc@1 86.498 Acc@5 98.026 loss 0.479
```

</details>

### fine-tuning


<details>
<summary>Evaluate the linear probing performance of EVA-CLIP vision encoder (<code>224px, patch_size=14</code>) on <b>ImageNet-1K val</b> with a single node (click to expand).</summary>

```bash    
MODEL_NAME=eva_g_patch14

sz=224
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_clip_vis_enc_sz224_ftcls_89p1.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz224_ftcls_89p1.pt

DATA_PATH=/path/to/ImageNet-1K/


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --dist_eval \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 89.074 Acc@5 98.710 loss 0.726
```

</details>



<details>
<summary>Evaluate the linear probing performance of EVA-CLIP vision encoder (<code>336px, patch_size=14</code>) on <b>ImageNet-1K val</b> with a single node (click to expand).</summary>

```bash    
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
crop_pct=1.0

EVAL_CKPT=/path/to/eva_clip_vis_enc_sz336_ftcls_89p4.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz336_ftcls_89p4.pt

DATA_PATH=/path/to/ImageNet-1K/


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --model ${MODEL_NAME} \
        --finetune ${EVAL_CKPT} \
        --input_size ${sz} \
        --batch_size ${batch_size} \
        --crop_pct ${crop_pct} \
        --no_auto_resume \
        --linear_probe \
        --eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 89.378 Acc@5 98.792 loss 0.691
```

</details>

## Pre-train **EVA** on the merged-30M image dataset

<details>
<summary>Structure of our merged-30M image dataset (click to expand)</summary>

```bash
merged_30m_pt
├── 21k
│   └── imagnet21k -> /path/to/ImageNet-21K
├── ade
│   └── training -> /path/to/ADEChallengeData2016/images/training
├── cc12m
│   └── pt_img_data -> /path/to/CC12M/pt_img_data
├── cc3m
│   └── train_image -> /path/to/cc-3m/conceptual-captions/train_image
├── coco
│   └── train2017 -> /path/to/coco/train2017
└── o365
    └── pt_images -> /path/to/Objects365/pt_images

```

</details>

<details>
<summary>We use 16 nodes (<code>total_bsz = 16*8*32 = 4096</code>) for pre-training (click to expand).</summary>

```bash
MODEL_NAME=eva_g_patch14

DATA_PATH=/path/to/merged_30m_pt
VAL_DATA_PATH=/path/to/ImageNet-1K # monitoring val loss 

input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4 ###

batch_size=32
update_freq=1

lr=1e-3
b2=0.98
eps=1e-6
dpr=0.1
ls=0.0

epochs=150
wmep=2

mixup=0.0
cj=0.0

zero_stage=1
save_ckpt_freq=1

teacher_type=clip
clip_model=ViT-L/14
cache_dir=/path/to/clip/large   # "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",


EXP_NAME=merge30M_${MODEL}_sz${input_size}_mask${num_mask_patches}_lr${lr}_b2${b2}_eps${eps}_dpr${dpr}_ls${ls}_bsz16x8x${batch_size}_ep${epochs}_wmep${wmep}_cj${cj}_ftpye${feature_type}_ltype${loss_type}_mixup${mixup}_abspos

OUTPUT_DIR=/path/to/output/${epochs}/${EXP_NAME}


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_eva_pretraining.py \
        --data_path ${DATA_PATH} \
        --val_data_path ${VAL_DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL} \
        --teacher_type ${teacher_type} \
        --clip_model ${clip_model} \
        --cache_dir ${cache_dir} \
        --input_size ${input_size} --second_input_size ${input_size} \
        --num_mask_patches ${num_mask_patches} \
        --layer_scale_init_value ${ls} \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --opt_betas 0.9 ${b2} \
        --opt_eps ${eps} \
        --drop_path ${dpr} \
        --epochs ${epochs} \
        --mixup ${mixup} \
        --color_jitter ${cj} \
        --warmup_epochs ${wmep} \
        --update_freq ${update_freq} \
        --clip_grad 3.0 \
        --weight_decay 0.05 \
        --rand \
        --zero_stage ${zero_stage} \
        --save_ckpt_freq ${save_ckpt_freq} \
        --enable_deepspeed

```

</details>


## Intermediate Fine-tune MIM pre-trained **EVA** on ImageNet-21K

<details>
<summary>We use 8 nodes (<code>total_bsz = 8*8*64 = 4096</code>) for intermediate fine-tuning (click to expand).</summary>

```bash

MODEL_NAME=eva_g_patch14

sz=224
batch_size=64
update_freq=1
lr=1e-4
lrd=0.85
partial_freeze=0
ep=60
wmep=15
reprob=0.0
dpr=0.4
mixup=0.0
cutmix=1.0
zero_stage=1
crop_pct=1.0
b2=0.98
eps=1e-6
scale_low=0.5

EXP_NAME=sz${sz}_cropscalelow${scale_low}_bsz8x8x${update_freq}x${batch_size}_lr${lr}_lrd${lrd}_b2${b2}_eps${eps}_partial_frz${partial_freeze}_ep${ep}_wmep${wmep}_reprob${reprob}_dpr${dpr}_mixup${mixup}_cutmix${cutmix}_crop_pct${crop_pct}

# path to MIM pre-trained ckpt
PRETRAIN_CHKPT=/path/to/eva_psz14.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_psz14.pt

OUTPUT_DIR=/path/to/output/{EXP_NAME}

DATA_PATH=/path/to/ImageNet-21K

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH} \
        --disable_eval_during_finetuning \
        --nb_classes 21841 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CHKPT} \
        --input_size ${sz} \
        --lr ${lr} \
        --layer_decay ${lrd} \
        --opt_betas 0.9 ${b2} \
        --opt_eps ${eps} \
        --epochs ${ep} \
        --warmup_epochs ${wmep} \
        --drop_path ${dpr} \
        --reprob ${reprob} \
        --mixup ${mixup} \
        --cutmix ${cutmix} \
        --batch_size ${batch_size} \
        --update_freq ${update_freq} \
        --crop_pct ${crop_pct} \
        --zero_stage ${zero_stage} \
        --partial_freeze ${partial_freeze} \
        --weight_decay 0.05 \
        --scale ${scale_low} 1.0 \
        --use_checkpoint \
        --enable_deepspeed

```

</details>



## Fine-tuning **EVA** on ImageNet-1K with ImageNet-21K intermediate fine-tuned checkpoint

<details>
<summary>We use 4 nodes (<code>total_bsz = 4*8*16 = 512</code>) for fine-tuning (click to expand).</summary>

```bash   
MODEL_NAME=eva_g_patch14

sz=336  # or 560
batch_size=16
update_freq=1

lr=3e-5      
lrd=0.95        

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=10   # or 15
wmep=2
dpr=0.4

reprob=0.0
mixup=0.0
cutmix=0.0

zero_stage=1
scale_low=0.08
crop_pct=1.0
smoothing=0.3
aa=rand-m9-mstd0.5-inc1


EXP_NAME=sz${sz}_cropscalelow${scale_low}_bsz4x8x${update_freq}x${batch_size}_lr${lr}_wmuplr${warmup_lr}_minlr${min_lr}_wd${weight_decay}_lrd${lrd}_partial_frz${partial_freeze}_ep${ep}_wmep${wmep}_reprob${reprob}_dpr${dpr}_mixup${mixup}_cutmix${cutmix}_aa${aa}_crop_pct${crop_pct}_sm${smoothing}


# path to ImageNet-21K Intermediate fine-tuned ckpt
PRETRAIN_CHKPT=/path/to/eva_21k_224px_psz14.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_21k_224px_psz14.pt

OUTPUT_DIR=/path/to/output/{EXP_NAME}

DATA_PATH=/path/to/ImageNet-1K


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CHKPT} \
        --input_size ${sz} \
        --scale ${scale_low} 1.0 \
        --lr ${lr} \
        --warmup_lr ${warmup_lr} \
        --min_lr ${min_lr} \
        --layer_decay ${lrd} \
        --epochs ${ep} \
        --warmup_epochs ${wmep} \
        --drop_path ${dpr} \
        --reprob ${reprob} \
        --mixup ${mixup} \
        --cutmix ${cutmix} \
        --batch_size ${batch_size} \
        --update_freq ${update_freq} \
        --crop_pct ${crop_pct} \
        --zero_stage ${zero_stage} \
        --partial_freeze ${partial_freeze} \
        --smoothing ${smoothing} \
        --weight_decay ${weight_decay} \
        --aa ${aa} \
        --dist_eval \
        --use_checkpoint \
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>


## Transferring **EVA-CLIP** vision encoder to ImageNet-1K 

### linear probing 


<details>
<summary>We use 5 nodes (<code>total_bsz = 5*8*400 = 16000</code>) for linear probing EVA-CLIP vision encoder w/ <code>224px</code> inputs (click to expand).</summary>

```bash   
MODEL_NAME=eva_g_patch14

sz=224 
batch_size=400
update_freq=1

lr=1.0      
lrd=1.0        

warmup_lr=0.0
min_lr=0.0
weight_decay=0.0

partial_freeze=0
ep=90
wmep=10
dpr=0.0

reprob=0.0
mixup=0.0
cutmix=0.0

zero_stage=0

scale_low=0.08
crop_pct=1.0
smoothing=0.0
aa=None


EXP_NAME=sz${sz}_cropscalelow${scale_low}_bsz4x8x${update_freq}x${batch_size}_lr${lr}_wmuplr${warmup_lr}_minlr${min_lr}_wd${weight_decay}_lrd${lrd}_partial_frz${partial_freeze}_ep${ep}_wmep${wmep}_reprob${reprob}_dpr${dpr}_mixup${mixup}_cutmix${cutmix}_aa${aa}_crop_pct${crop_pct}_sm${smoothing}


# path to EVA-CLIP vision encoder ckpt
PRETRAIN_CHKPT=/path/to/eva_clip_psz14_vision_enc.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14_vision_enc.pt

OUTPUT_DIR=/path/to/output/{EXP_NAME}

DATA_PATH=/path/to/ImageNet-1K


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CHKPT} \
        --input_size ${sz} \
        --scale ${scale_low} 1.0 \
        --lr ${lr} \
        --warmup_lr ${warmup_lr} \
        --min_lr ${min_lr} \
        --layer_decay ${lrd} \
        --epochs ${ep} \
        --warmup_epochs ${wmep} \
        --drop_path ${dpr} \
        --reprob ${reprob} \
        --mixup ${mixup} \
        --cutmix ${cutmix} \
        --batch_size ${batch_size} \
        --update_freq ${update_freq} \
        --crop_pct ${crop_pct} \
        --zero_stage ${zero_stage} \
        --partial_freeze ${partial_freeze} \
        --smoothing ${smoothing} \
        --weight_decay ${weight_decay} \
        --aa ${aa} \
        --dist_eval \
        --linear_probe \
        --use_cls
```

</details>



<details>
<summary>We use 5 nodes (<code>total_bsz = 5*8*400 = 16000</code>) for linear probing EVA-CLIP vision encoder w/ <code>336px</code> inputs (click to expand).</summary>

```bash   
MODEL_NAME=eva_g_patch14

sz=336
batch_size=400
update_freq=1

lr=0.6      
lrd=1.0        

warmup_lr=0.0
min_lr=0.0
weight_decay=0.0

partial_freeze=0
ep=90
wmep=10
dpr=0.0

reprob=0.0
mixup=0.0
cutmix=0.0

zero_stage=0

scale_low=0.08
crop_pct=1.0
smoothing=0.0
aa=None


EXP_NAME=sz${sz}_cropscalelow${scale_low}_bsz4x8x${update_freq}x${batch_size}_lr${lr}_wmuplr${warmup_lr}_minlr${min_lr}_wd${weight_decay}_lrd${lrd}_partial_frz${partial_freeze}_ep${ep}_wmep${wmep}_reprob${reprob}_dpr${dpr}_mixup${mixup}_cutmix${cutmix}_aa${aa}_crop_pct${crop_pct}_sm${smoothing}


# path to EVA-CLIP vision encoder ckpt
PRETRAIN_CHKPT=/path/to/eva_clip_psz14_vision_enc.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14_vision_enc.pt

OUTPUT_DIR=/path/to/output/{EXP_NAME}

DATA_PATH=/path/to/ImageNet-1K


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CHKPT} \
        --input_size ${sz} \
        --scale ${scale_low} 1.0 \
        --lr ${lr} \
        --warmup_lr ${warmup_lr} \
        --min_lr ${min_lr} \
        --layer_decay ${lrd} \
        --epochs ${ep} \
        --warmup_epochs ${wmep} \
        --drop_path ${dpr} \
        --reprob ${reprob} \
        --mixup ${mixup} \
        --cutmix ${cutmix} \
        --batch_size ${batch_size} \
        --update_freq ${update_freq} \
        --crop_pct ${crop_pct} \
        --zero_stage ${zero_stage} \
        --partial_freeze ${partial_freeze} \
        --smoothing ${smoothing} \
        --weight_decay ${weight_decay} \
        --aa ${aa} \
        --dist_eval \
        --linear_probe \
        --use_cls
```

</details>


### fine-tuning 


<details>
<summary>We use 4 nodes (<code>total_bsz = 4*8*32 = 1024</code>) for fine-tuning EVA-CLIP vision encoder w/ <code>224px</code> inputs (click to expand).</summary>

```bash   
MODEL_NAME=eva_g_patch14

sz=224 
batch_size=32
update_freq=1

lr=3e-5      
lrd=0.9        

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=20
wmep=2
dpr=0.4

reprob=0.0
mixup=0.0
cutmix=0.0

zero_stage=1
scale_low=0.08
crop_pct=1.0
smoothing=0.3
aa=rand-m9-mstd0.5-inc1


EXP_NAME=sz${sz}_cropscalelow${scale_low}_bsz4x8x${update_freq}x${batch_size}_lr${lr}_wmuplr${warmup_lr}_minlr${min_lr}_wd${weight_decay}_lrd${lrd}_partial_frz${partial_freeze}_ep${ep}_wmep${wmep}_reprob${reprob}_dpr${dpr}_mixup${mixup}_cutmix${cutmix}_aa${aa}_crop_pct${crop_pct}_sm${smoothing}


# path to EVA-CLIP vision encoder ckpt
PRETRAIN_CHKPT=/path/to/eva_clip_psz14_vision_enc.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14_vision_enc.pt

OUTPUT_DIR=/path/to/output/{EXP_NAME}

DATA_PATH=/path/to/ImageNet-1K


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CHKPT} \
        --input_size ${sz} \
        --scale ${scale_low} 1.0 \
        --lr ${lr} \
        --warmup_lr ${warmup_lr} \
        --min_lr ${min_lr} \
        --layer_decay ${lrd} \
        --epochs ${ep} \
        --warmup_epochs ${wmep} \
        --drop_path ${dpr} \
        --reprob ${reprob} \
        --mixup ${mixup} \
        --cutmix ${cutmix} \
        --batch_size ${batch_size} \
        --update_freq ${update_freq} \
        --crop_pct ${crop_pct} \
        --zero_stage ${zero_stage} \
        --partial_freeze ${partial_freeze} \
        --smoothing ${smoothing} \
        --weight_decay ${weight_decay} \
        --aa ${aa} \
        --dist_eval \
        --use_checkpoint \
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>




<details>
<summary>We use 4 nodes (<code>total_bsz = 4*8*16 = 512</code>) for fine-tuning EVA-CLIP vision encoder w/ <code>336px</code> inputs (click to expand).</summary>

```bash   
MODEL_NAME=eva_g_patch14

sz=336
batch_size=16
update_freq=1

lr=3e-5      
lrd=0.9        

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=20
wmep=2
dpr=0.4

reprob=0.0
mixup=0.0
cutmix=0.0

zero_stage=1
scale_low=0.08
crop_pct=1.0
smoothing=0.3
aa=rand-m9-mstd0.5-inc1


EXP_NAME=sz${sz}_cropscalelow${scale_low}_bsz4x8x${update_freq}x${batch_size}_lr${lr}_wmuplr${warmup_lr}_minlr${min_lr}_wd${weight_decay}_lrd${lrd}_partial_frz${partial_freeze}_ep${ep}_wmep${wmep}_reprob${reprob}_dpr${dpr}_mixup${mixup}_cutmix${cutmix}_aa${aa}_crop_pct${crop_pct}_sm${smoothing}


# path to EVA-CLIP vision encoder ckpt
PRETRAIN_CHKPT=/path/to/eva_clip_psz14_vision_enc.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14_vision_enc.pt

OUTPUT_DIR=/path/to/output/{EXP_NAME}

DATA_PATH=/path/to/ImageNet-1K


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CHKPT} \
        --input_size ${sz} \
        --scale ${scale_low} 1.0 \
        --lr ${lr} \
        --warmup_lr ${warmup_lr} \
        --min_lr ${min_lr} \
        --layer_decay ${lrd} \
        --epochs ${ep} \
        --warmup_epochs ${wmep} \
        --drop_path ${dpr} \
        --reprob ${reprob} \
        --mixup ${mixup} \
        --cutmix ${cutmix} \
        --batch_size ${batch_size} \
        --update_freq ${update_freq} \
        --crop_pct ${crop_pct} \
        --zero_stage ${zero_stage} \
        --partial_freeze ${partial_freeze} \
        --smoothing ${smoothing} \
        --weight_decay ${weight_decay} \
        --aa ${aa} \
        --dist_eval \
        --use_checkpoint \
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>


## Acknowledgement

This part of EVA is built using the awesome [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [BEiTv2](https://github.com/microsoft/unilm/tree/master/beit), [CLIP](https://github.com/openai/CLIP), [MAE](https://github.com/facebookresearch/mae/), [timm](https://github.com/rwightman/pytorch-image-models) and [DeepSpeed](https://github.com/microsoft/DeepSpeed) libraries.
Thanks for their wonderful works!
