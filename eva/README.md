# EVA: Pre-training and Image Classification 

**Table of Contents**

- [EVA: Pre-training and Image Classification](#eva-pre-training-and-image-classification)
  - [Model Card](#model-card)
  - [Summary of EVA's image classification performance](#summary-of-evas-image-classification-performance)
  - [Setup](#setup)
  - [Evaluate EVA on ImageNet-1K](#evaluate-eva-on-imagenet-1k)
<<<<<<< HEAD
  - [Evaluation on ImageNet-1K variants](#evaluation-on-imagenet-1k-variants)
=======
  - [Evaluate EVA on ImageNet-1K variants (ImageNet-V2/ReaL/Adv/Rendition/Sketch)](#evaluation-on-imagenet-1k-variants)
>>>>>>> f35d528f764e749307ac3c5973a72d73158f6093
  - [Pre-train EVA on the merged-30M image dataset](#pre-train-eva-on-the-merged-30m-image-dataset)
  - [Intermediate Fine-tune MIM pre-trained EVA on ImageNet-21K](#intermediate-fine-tune-mim-pre-trained-eva-on-imagenet-21k)
  - [Fine-tuning EVA on ImageNet-1K with ImageNet-21K intermediate fine-tuned checkpoint](#fine-tuning-eva-on-imagenet-1k-with-imagenet-21k-intermediate-fine-tuned-checkpoint)
  - [Acknowledgement](#acknowledgement)

## Model Card


We provide **all pre-trained & fine-tuned** EVAs for the community. 
The following table summarizes the basic statistics of MIM pre-trained EVA and image classification EVA.

| model name | #param. |pre-training epochs on merged-30M | intermeidate fine-tuning epochs on IN-21K | fine-tuning epochs on IN-1K | IN-1K top-1 acc. |weight |
|------------|:------:|:------------------:|:------:|:------:|:------:|:------:|
| `eva_psz14` | 1.0B | 150 | - | - | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14.pt) (`2GB`) |
| `eva_psz14to16` | 1.0B | 150 | - | - | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt) (`2GB`) | 
| `eva_21k_224px_psz14` | 1.0B | 150 | 60 | - | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_21k_224px_psz14.pt) (`2GB`) |
| `eva_21k_1k_336px_psz14_ema` | 1.0B | 150 | 60 | 10 | 89.6 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt) (`4GB`) |
| `eva_21k_1k_560px_psz14_ema` | 1.0B | 150 | 60 | 15 | 89.7 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_560px_psz14_ema_89p7.pt) (`4GB`) |

- `eva_psz14to16` model interpolates the kernel size of `patch_embed` from `14x14` to `16x16`. This is useful for object detection, instance segmentation & semantic segmentation, *etc*. See [`interpolate_patch_14to16.py`](interpolate_patch_14to16.py) for implementation details.
- For MIM pre-trained EVA and EVA-CLIP, we use `deepspeed` `fp16` format. IN-1K fine-tuned EVA weights are larger (`4GB` *v.s.* `2GB`) because ema updates models with `fp32` format. The weights of other downstream tasks are also with `fp32` format.




## Summary of EVA's image classification performance

<div align="center">

| model | [IN-1K](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet.csv) | [IN-V2](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv) | [IN-ReaL](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-real.csv) | [IN-Adv.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-a.csv) | [IN-Ren.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv) | [IN-Ske.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv) | ObjectNet |
|:------------:|:------------------:|:------:|:------:| :------:|:------:|:------:|:------:|
| EVA | 89.6 | 81.6 | 90.8 | 86.2 | 88.3 | 67.7 | 60.9 |

</div>

> The top-1 accuracy of ImageNet-1K variants are better than the results we reported in our paper for we fix a bug. We will update the new results in the revision soon. 

For reference, [timm](https://github.com/rwightman/pytorch-image-models) collects some open-sourced state-of-the-art models' image classification results [here](https://github.com/rwightman/pytorch-image-models/tree/main/results) ([IN-1K](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet.csv), [IN-V2](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv), [IN-ReaL](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-real.csv), [IN-Adv.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-a.csv), [IN-Ren.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv), [IN-Ske.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv)).

Compared with other open-sourced models, EVA achieves the state-of-the-art performance in all the classification benchmarks we evaluated. 

For zero-shot classification performance of EVA-CLIP, please refer to [`clip`](clip) and [wandb logs](https://wandb.ai/baaivision/eva-clip/reports/ViT-g-14--VmlldzoyOTkwMDYy).


## Setup


First, clone the repo and install required packages:
```
git clone git@github.com:baaivision/EVA.git
cd pt_and_img_cls
pip install -r requirements.txt
```

The core packages including: [Pytorch](https://pytorch.org/) version 1.12.0, [torchvision](https://pytorch.org/vision/stable/index.html) version 0.13.0, [timm](https://github.com/rwightman/pytorch-image-models) version 0.5.4 and [DeepSpeed](https://github.com/microsoft/DeepSpeed) version 0.7.5 *etc*.




## Evaluate EVA on ImageNet-1K

Evaluate the fine-tuned EVA (`336px, patch_size=14`) on **ImageNet-1K val** with a single node:
```bash    

MODEL_NAME=eva_g_patch14

sz=448
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

Evaluate the fine-tuned EVA (`560px, patch_size=14`) on **ImageNet-1K val** with a single node:
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
* * Acc@1 89.712 Acc@5 98.958 loss 0.881
```

## Evaluation on ImageNet-1K variants


Evaluate the fine-tuned EVA (`336px, patch_size=14`) on **ImageNet-V2** with a single node:
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



Evaluate the fine-tuned EVA (`336px, patch_size=14`) on **ImageNet-ReaL** with a single GPU on a single node:
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



Evaluate the fine-tuned EVA (`336px, patch_size=14`) on **ImageNet-Adversarial** with a single node:
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




Evaluate the fine-tuned EVA (`336px, patch_size=14`) on **ImageNet-Rendition** with a single node:
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



Evaluate the fine-tuned EVA (`336px, patch_size=14`) on **ImageNet-Sketch** with a single node:
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



Evaluate the fine-tuned EVA (`336px, patch_size=14`) on **ObjectNet** with a single node:
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
* * Acc@1 60.907 Acc@5 82.768 loss 2.305
```


## Pre-train EVA on the merged-30M image dataset

Structure of our merged-30M image dataset:

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

We use 16 nodes (`total_bsz = 16*8*32 = 4096`) for pre-training.

```bash

MODEL_NAME=eva_g_patch14

DATA_PATH=/path/to/merged_30m_pt

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
--master_addr=$MASTER_ADDR --master_port=12355 --use_env run_beit_pretraining.py \
        --data_path ${DATA_PATH} \
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


## Intermediate Fine-tune MIM pre-trained EVA on ImageNet-21K

We use 8 nodes (`total_bsz = 8*8*64 = 4096`) for intermediate fine-tuning.

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


## Fine-tuning EVA on ImageNet-1K with ImageNet-21K intermediate fine-tuned checkpoint


We use 4 nodes (`total_bsz = 4*8*16 = 512`) for fine-tuning.

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




## Acknowledgement

This part of EVA is built using the awesome [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [BEiTv2](https://github.com/microsoft/unilm/tree/master/beit), [CLIP](https://github.com/openai/CLIP), [MAE](https://github.com/facebookresearch/mae/), [timm](https://github.com/rwightman/pytorch-image-models) and [DeepSpeed](https://github.com/microsoft/DeepSpeed) libraries.
Thanks for their wonderful works!
