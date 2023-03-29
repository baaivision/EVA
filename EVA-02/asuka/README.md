# EVA-02: MIM Pre-training and Image Classification 

**Table of Contents**

- [EVA-02: MIM Pre-training and Image Classification](#eva-02-mim-pre-training-and-image-classification)
  - [EVA-02 Model Card](#eva-02-model-card)
    - [MIM pre-trained EVA-02](#mim-pre-trained-eva-02)
    - [IN-21K intermediate fine-tuned EVA-02](#in-21k-intermediate-fine-tuned-eva-02)
    - [IN-1K fine-tuned EVA-02 (*w/o* IN-21K intermediate fine-tuning)](#in-1k-fine-tuned-eva-02-wo-in-21k-intermediate-fine-tuning)
    - [IN-1K fine-tuned EVA-02 (*w/* IN-21K intermediate fine-tuning)](#in-1k-fine-tuned-eva-02-w-in-21k-intermediate-fine-tuning)
    - [Summary of EVA-02 performance on IN-1K val and variants](#summary-of-eva-02-performance-on-in-1k-val-and-variants)
  - [Setup](#setup)
  - [Evaluation of Image Classification Performance](#evaluation-of-image-classification-performance)
    - [Evaluate Fine-tuned EVA-02 on IN-1K](#evaluate-fine-tuned-eva-02-on-in-1k)
      - [w/o IN-21K intermediate fine-tuning (MIM -\> IN-1K)](#wo-in-21k-intermediate-fine-tuning-mim---in-1k)
      - [w/ IN-21K intermediate fine-tuning (MIM -\> IN-21K -\> IN-21K)](#w-in-21k-intermediate-fine-tuning-mim---in-21k---in-21k)
    - [Evaluate EVA-02 on IN-1K variants (IN-V2, IN-ReaL, IN-Adv., IN-Ren., IN-Ske., ObjectNet)](#evaluate-eva-02-on-in-1k-variants-in-v2-in-real-in-adv-in-ren-in-ske-objectnet)
  - [Pre-training](#pre-training)
    - [Pre-train EVA-02 on IN-21K unlabeled image dataset](#pre-train-eva-02-on-in-21k-unlabeled-image-dataset)
  - [Fine-tuning](#fine-tuning)
    - [Directly Fine-tune MIM pre-trained EVA-02 on IN-1K](#directly-fine-tune-mim-pre-trained-eva-02-on-in-1k)
    - [Intermediate Fine-tune MIM pre-trained EVA-02 on IN-21K](#intermediate-fine-tune-mim-pre-trained-eva-02-on-in-21k)
    - [Fine-tune EVA-02 on IN-1K with IN-21K intermediate fine-tuned checkpoint](#fine-tune-eva-02-on-in-1k-with-in-21k-intermediate-fine-tuned-checkpoint)
  - [Acknowledgement](#acknowledgement)


## EVA-02 Model Card


### MIM pre-trained EVA-02

<div align="center">

| model name | #params | MIM pt dataset | MIM pt epochs | log | weight |
|------------|:-------:|:--------------:|:-------------:|:---:|:------:|
| `eva02_Ti_pt_in21k_p14` | 6M | IN-21K | 240 | [link](https://gist.github.com/Yuxin-CV/6491f01a3a7a2f31115fb7a7a19c7148#file-eva02_ti_pt_in21k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_Ti_pt_in21k_p14.pt) |
| `eva02_S_pt_in21k_p14` | 22M | IN-21K | 240 | [link](https://gist.github.com/Yuxin-CV/6491f01a3a7a2f31115fb7a7a19c7148#file-eva02_s_pt_in21k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_S_pt_in21k_p14.pt) |
| `eva02_B_pt_in21k_p14` | 86M | IN-21K | 150 | [link](https://gist.github.com/Yuxin-CV/6491f01a3a7a2f31115fb7a7a19c7148#file-eva02_b_pt_in21k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_B_pt_in21k_p14.pt) |
| `eva02_B_pt_in21k_p14to16` | 86M | IN-21K | 150 | - | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_B_pt_in21k_p14to16.pt) |
| `eva02_L_pt_in21k_p14` | 304M | IN-21K | 150 | [link](https://gist.github.com/Yuxin-CV/6491f01a3a7a2f31115fb7a7a19c7148#file-eva02_l_pt_in21k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_L_pt_in21k_p14.pt) |
| `eva02_L_pt_in21k_p14to16` | 304M | IN-21K | 150 | - | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_L_pt_in21k_p14to16.pt) |
| `eva02_L_pt_m38m_p14` | 304M | Merged-38M | 56 | [link](https://gist.github.com/Yuxin-CV/6491f01a3a7a2f31115fb7a7a19c7148#file-eva02_l_pt_m38m_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_L_pt_m38m_p14.pt) |
| `eva02_L_pt_m38m_p14to16` | 304M | Merged-38M | 56 | - | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_L_pt_m38m_p14to16.pt) |

</div>

- The input size / patch size of MIM pre-trained EVA-02 is `224x224` / `14x14`.
- `eva02_psz14to16` models interpolate the kernel size of `patch_embed` from `14x14` to `16x16`, and interpolate the `pos_embed` from `16x16` to `14x14`. This is useful for object detection, instance segmentation & semantic segmentation tasks.


### IN-21K intermediate fine-tuned EVA-02

<div align="center">

| model name | init. ckpt | IN-21K ft epochs | log | weight |
|------------|------|:------:|:------:|:------:|
| `eva02_B_pt_in21k_medft_in21k_p14` | `eva02_B_pt_in21k_p14` | 40 | [link](https://gist.github.com/Yuxin-CV/6761e2564d13e497826ace491017fe4a#file-eva02_b_pt_in21k_medft_in21k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in21k/eva02_B_pt_in21k_medft_in21k_p14.pt) |
| `eva02_L_pt_in21k_medft_in21k_p14` | `eva02_L_pt_in21k_p14` | 20 | [link](https://gist.github.com/Yuxin-CV/6761e2564d13e497826ace491017fe4a#file-eva02_l_pt_in21k_medft_in21k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in21k/eva02_L_pt_in21k_medft_in21k_p14.pt) |
| `eva02_L_pt_m38m_medft_in21k_p14` | `eva02_L_pt_m38m_p14` | 30 | [link](https://gist.github.com/Yuxin-CV/6761e2564d13e497826ace491017fe4a#file-eva02_l_pt_m38m_medft_in21k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in21k/eva02_L_pt_m38m_medft_in21k_p14.pt) |

</div>

- The input size / patch size of IN-21K intermediate fine-tuned EVA-02 is `448x448` / `14x14`.



### IN-1K fine-tuned EVA-02 (*w/o* IN-21K intermediate fine-tuning) 

<div align="center">

| model name | init. ckpt | IN-1K ft epochs | ft image size | ema？| top-1 | log | weight |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `eva02_Ti_pt_in21k_ft_in1k_p14` | `eva02_Ti_pt_in21k_p14` | 100 | ``336x336`` | `x` | 80.7 |[link](https://gist.github.com/Yuxin-CV/308957078ef8eaf361a42d57bd211f22#file-eva02_ti_pt_in21k_ft_in1k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_Ti_pt_in21k_ft_in1k_p14.pt) |
| `eva02_S_pt_in21k_ft_in1k_p14` | `eva02_S_pt_in21k_p14` | 100 | ``336x336`` | `x` | 85.8 | [link](https://gist.github.com/Yuxin-CV/308957078ef8eaf361a42d57bd211f22#file-eva02_s_pt_in21k_ft_in1k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_S_pt_in21k_ft_in1k_p14.pt) |
| `eva02_B_pt_in21k_ft_in1k_p14` | `eva02_B_pt_in21k_p14` | 30 | ``448x448`` | `x` | 88.3 | [link](https://gist.github.com/Yuxin-CV/308957078ef8eaf361a42d57bd211f22#file-eva02_b_pt_in21k_ft_in1k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_B_pt_in21k_ft_in1k_p14.pt) |
| `eva02_L_pt_in21k_ft_in1k_p14` | `eva02_L_pt_in21k_p14` | 30 | ``448x448`` | `o`| 89.6 | [link](https://gist.github.com/Yuxin-CV/308957078ef8eaf361a42d57bd211f22#file-eva02_l_pt_in21k_ft_in1k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_L_pt_in21k_ft_in1k_p14.pt) |
| `eva02_L_pt_m38m_ft_in1k_p14` | `eva02_L_pt_m38m_p14` | 30 | `448x448` | `o` | 89.6 | [link](https://gist.github.com/Yuxin-CV/308957078ef8eaf361a42d57bd211f22#file-eva02_l_pt_m38m_ft_in1k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_L_pt_m38m_ft_in1k_p14.pt) |

</div>

- `o`: using ema model weight update achieves similar or slightly improved performance.



### IN-1K fine-tuned EVA-02 (*w/* IN-21K intermediate fine-tuning) 

<div align="center">

| model name | init. ckpt | IN-1K ft epochs | ft image size | ema？| top-1 | log | weight |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `eva02_B_pt_in21k_medft_in21k_ft_in1k_p14` | `eva02_B_pt_in21k_medft_in21k_p14` | 15 | ``448x448`` | `o` | 88.6 | [link](https://gist.github.com/Yuxin-CV/0a52be2ee1730636a64743367f20e8d6#file-eva02_b_pt_in21k_medft_in21k_ft_in1k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in21k_to_in1k/eva02_B_pt_in21k_medft_in21k_ft_in1k_p14.pt) |
| `eva02_L_pt_in21k_medft_in21k_ft_in1k_p14` | `eva02_L_pt_in21k_medft_in21k_p14` | 20 | ``448x448`` | `o` | 89.9 | [link](https://gist.github.com/Yuxin-CV/0a52be2ee1730636a64743367f20e8d6#file-eva02_l_pt_in21k_medft_in21k_ft_in1k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in21k_to_in1k/eva02_L_pt_in21k_medft_in21k_ft_in1k_p14.pt) |
| `eva02_L_pt_m38m_medft_in21k_ft_in1k_p14` | `eva02_L_pt_m38m_medft_in21k_p14` | 20 | `448x448` | `o` | 90.0 | [link](https://gist.github.com/Yuxin-CV/0a52be2ee1730636a64743367f20e8d6#file-eva02_l_pt_m38m_medft_in21k_ft_in1k_p14) | [🤗 HF link](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in21k_to_in1k/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt) |

</div>

- `o`: using ema model weight update achieves similar or slightly improved performance.




### Summary of EVA-02 performance on IN-1K val and variants 

<div align="center">

| model name | [IN-1K](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet.csv) | [IN-V2](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv) | [IN-ReaL](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-real.csv) | [IN-Adv.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-a.csv) | [IN-Ren.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv) | [IN-Ske.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv) | ObjectNet |
|--------|:------------------:|:------:|:------:| :------:|:------:|:------:|:------:|
| `eva02_B_pt_in21k_medft_in21k_ft_in1k_p14` | 88.6 | 79.8 | 90.8 | 78.1 | 76.8 | 57.7 | 55.3 |
| `eva02_L_pt_m38m_medft_in21k_ft_in1k_p14` | 90.0 | 82.4 | 91.1 | 87.7 | 89.9 | 70.1 | 62.8 |

</div>

For reference, [`timm`](https://github.com/rwightman/pytorch-image-models) collects some open-sourced state-of-the-art models' image classification results at [here](https://github.com/rwightman/pytorch-image-models/tree/main/results) ([IN-1K](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet.csv), [IN-V2](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv), [IN-ReaL](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-real.csv), [IN-Adv.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-a.csv), [IN-Ren.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv), [IN-Ske.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv)).





## Setup


First, clone the repo and install required packages:
```bash
conda create --name asuka python=3.8 -y
conda activate asuka

git clone git@github.com:baaivision/EVA.git
cd EVA-02/asuka
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

Then, install [Apex](https://github.com/NVIDIA/apex#linux) and [xFormer](https://github.com/facebookresearch/xformers#installing-xformers) following the official instruction. 


Core packages: 
- [Pytorch](https://pytorch.org/) version 1.12.1 
- [torchvision](https://pytorch.org/vision/stable/index.html) version 0.13.0
- [timm](https://github.com/rwightman/pytorch-image-models) version 0.5.4 
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) version 0.6.5 (`fp16` training and ZeRO optimizer), fine-tuning with `bfloat16` requires version 0.8.1
- [Apex](https://github.com/NVIDIA/apex) (fused layer norm)
- [xFormer](https://github.com/facebookresearch/xformers) (fast and memory efficient MHSA)




## Evaluation of Image Classification Performance

### Evaluate Fine-tuned EVA-02 on IN-1K


We use the standard IN-1K dataset (1.2M images). 
Download it from http://image-net.org.
Then, move and extract the training and validation images to labeled subfolders, using the [shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh).

#### w/o IN-21K intermediate fine-tuning (MIM -> IN-1K)
<details>
  <summary>Evaluate the fine-tuned <code>eva02_Ti_pt_in21k_ft_in1k_p14</code> on <b>IN-1K val</b> using a single node with 4 gpus (click to expand).</summary>

```bash    
MODEL_NAME=eva02_tiny_patch14_xattn_fusedLN_SwiGLU_preln_RoPE

sz=336
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_Ti_pt_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-1K/

# using model w/o ema for evaluation (w/o --use_ema_ckpt_eval)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
* Acc@1 80.714 Acc@5 95.536 loss 0.807
```

</details>





<details>
  <summary>Evaluate the fine-tuned <code>eva02_S_pt_in21k_ft_in1k_p14</code> on <b>IN-1K val</b> using a single node with 4 gpus (click to expand).</summary>

```bash    
MODEL_NAME=eva02_small_patch14_xattn_fusedLN_SwiGLU_preln_RoPE

sz=336
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_S_pt_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-1K/

# using model w/o ema for evaluation (w/o --use_ema_ckpt_eval)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
* Acc@1 85.780 Acc@5 97.598 loss 0.612
```

</details>






<details>
  <summary>Evaluate the fine-tuned <code>eva02_B_pt_in21k_ft_in1k_p14</code> on <b>IN-1K val</b> using a single node with 4 gpus (click to expand).</summary>

```bash    
MODEL_NAME=eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_B_pt_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-1K/

# using model w/o ema for evaluation (w/o --use_ema_ckpt_eval)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
* Acc@1 88.282 Acc@5 98.528 loss 0.507 
```

</details>






<details>
  <summary>Evaluate the fine-tuned <code>eva02_L_pt_in21k_ft_in1k_p14</code> on <b>IN-1K val</b> using a single node with 4 gpus (click to expand).</summary>

```bash    
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-1K/

# using model w/ ema for evaluation (w/ --use_ema_ckpt_eval)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 89.626 Acc@5 98.954 loss 0.599
```

</details>







<details>
  <summary>Evaluate the fine-tuned <code>eva02_L_pt_m38m_ft_in1k_p14</code> on <b>IN-1K val</b> with using a single node with 4 gpus (click to expand).</summary>

```bash    
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_m38m_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-1K/

# using model w/ ema for evaluation (w/ --use_ema_ckpt_eval)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 89.570 Acc@5 98.924 loss 0.612
```

</details>






#### w/ IN-21K intermediate fine-tuning (MIM -> IN-21K -> IN-21K)
<details>
  <summary>Evaluate the fine-tuned <code>eva02_B_pt_in21k_medft_in21k_ft_in1k_p14</code> on <b>IN-1K val</b> using a single node with 4 gpus (click to expand).</summary>

```bash    
MODEL_NAME=eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_B_pt_in21k_medft_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-1K/

# using model w/ ema for evaluation (w/ --use_ema_ckpt_eval)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 88.570 Acc@5 98.650 loss 0.686
```

</details>








<details>
  <summary>Evaluate the fine-tuned <code>eva02_L_pt_in21k_medft_in21k_ft_in1k_p14</code> on <b>IN-1K val</b> using a single node with 4 gpus (click to expand).</summary>

```bash    
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_in21k_medft_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-1K/

# using model w/ ema for evaluation (w/ --use_ema_ckpt_eval)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 89.904 Acc@5 98.974 loss 0.647
```

</details>








<details>
  <summary>Evaluate the fine-tuned <code>eva02_L_pt_m38m_medft_in21k_ft_in1k_p14</code> on <b>IN-1K val</b> using a single node with 4 gpus (click to expand).</summary>

```bash    
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-1K/

# using model w/ ema for evaluation (w/ --use_ema_ckpt_eval)
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 89.974 Acc@5 99.022 loss 0.700
```

</details>









### Evaluate EVA-02 on IN-1K variants (IN-V2, IN-ReaL, IN-Adv., IN-Ren., IN-Ske., ObjectNet)

We provide the evaluation instructions of ``eva02_L_pt_m38m_medft_in21k_ft_in1k_p14``. Evaluation of other EVA-02 models are similar.

Please download / prepare the IN-1K variants data from the official release first.


<details>
 <summary>Evaluate the fine-tuned <code>eva02_L_pt_m38m_medft_in21k_ft_in1k_p14</code> on <b>ImageNet-V2 (IN-V2)</b> using a single node with 4 gpus (click to expand).</summary>

```bash     
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-V2/ImageNetV2-matched-frequency


python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 82.430 Acc@5 96.360 loss 1.027
```

</details>




<details>
<summary>Evaluate the fine-tuned <code>eva02_L_pt_m38m_medft_in21k_ft_in1k_p14</code> on <b>ImageNet-ReaL (IN-ReaL)</b> using a single node with 1 gpu (click to expand).</summary>

```bash     
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-1K


python -m torch.distributed.launch --nproc_per_node=1 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* ReaL Acc@1 91.075 Acc@5 98.689 loss 0.699
```

</details>





<details>

<summary>Evaluate the fine-tuned <code>eva02_L_pt_m38m_medft_in21k_ft_in1k_p14</code> on <b>ImageNet-Adversarial (IN-Adv.)</b> using a single node with 4 gpus (click to expand).</summary>

```bash     
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-Adv

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 87.720 Acc@5 96.893 loss 0.829
```

</details>






<details>
<summary>Evaluate the fine-tuned <code>eva02_L_pt_m38m_medft_in21k_ft_in1k_p14</code> on <b>ImageNet-Rendition (IN-Ren.)</b> using a single node with 4 gpus (click to expand).</summary>


```bash     
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-Ren

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 89.907 Acc@5 96.957 loss 0.802
```

</details>





<details>
<summary>Evaluate the fine-tuned <code>eva02_L_pt_m38m_medft_in21k_ft_in1k_p14</code> on <b>ImageNet-Sketch (IN-Ske.)</b> using a single node with 4 gpus (click to expand).</summary>

```bash     
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt

DATA_PATH=/path/to/IN-Ske


python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 70.131 Acc@5 89.617 loss 1.647
```

</details>





<details>
<summary>Evaluate the fine-tuned <code>eva02_L_pt_m38m_medft_in21k_ft_in1k_p14</code> on <b>ObjectNet (ObjNet)</b> using a single node with 4 gpus (click to expand).</summary>

```bash     
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

sz=448
batch_size=64
crop_pct=1.0

EVAL_CKPT=/path/to/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt

DUMMY_DATA_PATH=/path/to/IN-1K
DATA_PATH=/path/to/ObjNet/objectnet-1.0/images

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
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
        --use_ema_ckpt_eval \
        --enable_deepspeed
```

Expected results:
```
* Acc@1 62.801 Acc@5 84.636 loss 2.002
```

</details>






## Pre-training

### Pre-train EVA-02 on IN-21K unlabeled image dataset

We provide instruction of pre-training EVA-02 on IN-21K dataset (14.2M images) and Merged-38M dataset. 

Please prepare IN-21K dataset, Merged-38M dataset and EVA-CLIP (`eva_clip_psz14.pt`, [download link](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt)) first.


<details>
<summary>Pre-train <code>eva02_Ti_pt_in21k_p14</code> on <b>IN-21K</b> using 5 nodes x 8 gpus per node (click to expand).</summary>

```bash
MODEL=eva02_tiny_patch14_xattn_fusedLN_SwiGLU_preln_RoPE_xavier_normal_init

DATA_PATH=/path/to/IN-21K
VAL_DATA_PATH=/path/to/IN-1K # monitoring val loss 

input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4 

batch_size=100  # 100(bsz_per_gpu)*8(#gpus_per_node)*5(#nodes)*1(update_freq)=4000(total_bsz)
update_freq=1

lr=3e-3
b2=0.98
eps=1e-6

dpr=0.0
ls=0.0

epochs=240
wmep=1
save_ckpt_freq=10

mixup=0.0
cj=0.0

zero_stage=0

teacher_type=evaclip
clip_model=EVA_CLIP_g_14_X
cache_dir=/path/to/eva_clip_psz14.pt


OUTPUT_DIR=/path/to/output/${MODEL}


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_eva02_pretraining.py \
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
        --weight_decay 0.05 \
        --zero_stage ${zero_stage} \
        --save_ckpt_freq ${save_ckpt_freq} \
        --stop_grad_conv1 \
        --enable_deepspeed

```

</details>











<details>
<summary>Pre-train <code>eva02_S_pt_in21k_p14</code> on <b>IN-21K</b> using 5 nodes x 8 gpus per node (click to expand).</summary>

```bash
MODEL=eva02_small_patch14_xattn_fusedLN_SwiGLU_preln_RoPE_xavier_normal_init

DATA_PATH=/path/to/IN-21K
VAL_DATA_PATH=/path/to/IN-1K # monitoring val loss 

input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4 

batch_size=100  # 100(bsz_per_gpu)*8(#gpus_per_node)*5(#nodes)*1(update_freq)=4000(total_bsz)
update_freq=1

lr=3e-3
b2=0.98
eps=1e-6

dpr=0.0
ls=0.0

epochs=240
wmep=1
save_ckpt_freq=10

mixup=0.0
cj=0.0

zero_stage=0

teacher_type=evaclip
clip_model=EVA_CLIP_g_14_X
cache_dir=/path/to/eva_clip_psz14.pt


OUTPUT_DIR=/path/to/output/${MODEL}


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_eva02_pretraining.py \
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
        --weight_decay 0.05 \
        --zero_stage ${zero_stage} \
        --save_ckpt_freq ${save_ckpt_freq} \
        --stop_grad_conv1 \
        --enable_deepspeed

```

</details>










<details>
<summary>Pre-train <code>eva02_B_pt_in21k_p14</code> on <b>IN-21K</b> using 4 nodes x 8 gpus per node (click to expand).</summary>

```bash
MODEL=eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_normal_init

DATA_PATH=/path/to/IN-21K
VAL_DATA_PATH=/path/to/IN-1K # monitoring val loss 

input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4 

batch_size=64  # 64(bsz_per_gpu)*8(#gpus_per_node)*4(#nodes)*1(update_freq)=2048(total_bsz)
update_freq=1

lr=1.5e-3
b2=0.98
eps=1e-6

dpr=0.0
ls=0.0

epochs=150
wmep=1
save_ckpt_freq=10

mixup=0.0
cj=0.0

zero_stage=0

teacher_type=evaclip
clip_model=EVA_CLIP_g_14_X
cache_dir=/path/to/eva_clip_psz14.pt


OUTPUT_DIR=/path/to/output/${MODEL}


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_eva02_pretraining.py \
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
        --weight_decay 0.05 \
        --zero_stage ${zero_stage} \
        --save_ckpt_freq ${save_ckpt_freq} \
        --stop_grad_conv1 \
        --enable_deepspeed

```

</details>






<details>
<summary>Pre-train <code>eva02_L_pt_in21k_p14</code> on <b>IN-21K</b> using 8 nodes x 8 gpus per node (click to expand).</summary>

```bash
MODEL=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_normal_init

DATA_PATH=/path/to/IN-21K
VAL_DATA_PATH=/path/to/IN-1K # monitoring val loss 

input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4 

batch_size=32  # 32(bsz_per_gpu)*8(#gpus_per_node)*8(#nodes)*1(update_freq)=2048(total_bsz)
update_freq=1

lr=1.5e-3
b2=0.98
eps=1e-6

dpr=0.1
ls=0.0

epochs=150
wmep=1
save_ckpt_freq=10

mixup=0.0
cj=0.0

zero_stage=1

teacher_type=evaclip
clip_model=EVA_CLIP_g_14_X
cache_dir=/path/to/eva_clip_psz14.pt


OUTPUT_DIR=/path/to/output/${MODEL}


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_eva02_pretraining.py \
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
        --weight_decay 0.05 \
        --zero_stage ${zero_stage} \
        --save_ckpt_freq ${save_ckpt_freq} \
        --stop_grad_conv1 \
        --enable_deepspeed

```

</details>





<details>
<summary>Pre-train <code>eva02_L_pt_m38m_p14</code> on <b>Merged-38M</b> using 8 nodes x 8 gpus per node (click to expand).</summary>

Prepare Merged-38M unlabeled image dataset:
```bash
Merged-38M
├── IN-21K
│   └── IN-21K -> /path/to/IN-21K
├── ADE20K
│   └── training -> /path/to/ADEChallengeData2016/images/training
├── CC12M
│   └── train_image -> /path/to/CC12M/train_image
├── CC3M
│   └── train_image -> /path/to/CC3M/train_image
├── COCO
│   └── train2017 -> /path/to/COCO/train2017
├── Object365
│   └── images -> /path/to/Objects365/images
└── OpenImages
    └── OpenImages_v6 -> /path/to/openimages_v6

```

Pre-training on Merged-38M unlabeled image dataset:
```bash
MODEL=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_normal_init

DATA_PATH=/path/to/Merged-38M
VAL_DATA_PATH=/path/to/IN-1K # monitoring val loss 

input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4 

batch_size=32  # 32(bsz_per_gpu)*8(#gpus_per_node)*8(#nodes)*1(update_freq)=2048(total_bsz)
update_freq=1

lr=1.5e-3
b2=0.98
eps=1e-6

dpr=0.1
ls=0.0

epochs=56
wmep=1
save_ckpt_freq=10

mixup=0.0
cj=0.0

zero_stage=1

teacher_type=evaclip
clip_model=EVA_CLIP_g_14_X
cache_dir=/path/to/eva_clip_psz14.pt


OUTPUT_DIR=/path/to/output/${MODEL}


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_eva02_pretraining.py \
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
        --weight_decay 0.05 \
        --zero_stage ${zero_stage} \
        --save_ckpt_freq ${save_ckpt_freq} \
        --stop_grad_conv1 \
        --enable_deepspeed

```

</details>




## Fine-tuning 

- By default, we fine-tune EVA-02 with `deepspeed==0.6.5` & `fp16`. Fine-tuning with `bfloat16` requires `deepspeed==0.8.1`. 
- If you receive complaints on **`size mismatch of RoPE`** when loading some pre-trained EVA-02 checkpoints, just ignore them. This is because previously we used a naive implementation [`VisionRotaryEmbedding`](https://github.com/baaivision/EVA/blob/8e966a91a9dbf60a0a96e6a6a2a9aa275a676907/EVA-02/asuka/rope.py#L46) for pre-training, and later we changed to a slightly faster & neater one [`VisionRotaryEmbeddingFast`](https://github.com/baaivision/EVA/blob/8e966a91a9dbf60a0a96e6a6a2a9aa275a676907/EVA-02/asuka/rope.py#L96). The only difference is they come with different RoPE shapes. Functionally they are the same. Also see https://github.com/baaivision/EVA/issues/56 if you have trouble loading EVA-02 MIM pre-trained weights.
  


### Directly Fine-tune MIM pre-trained EVA-02 on IN-1K



<details>
<summary>Fine-tune MIM pre-trained <code>eva02_Ti_pt_in21k_p14</code> on <b>IN-1K</b> using 1 nodes x 8 gpus per node (click to expand).</summary>

```bash   
MODEL_NAME=eva02_tiny_patch14_xattn_fusedLN_SwiGLU_preln_RoPE

PRETRAIN_CKPT=/path/to/eva02_Ti_pt_in21k_p14.pt 

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-1K


sz=336
batch_size=128  # 128(bsz_per_gpu)*8(#gpus_per_node)*1(#nodes)*1(update_freq)=1024(total_bsz)
update_freq=1

lr=2e-4           
lrd=0.9          

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=100
wmep=5
dpr=0.1

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.1

zero_stage=0

scale_low=0.08
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
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
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>







<details>
<summary>Fine-tune MIM pre-trained <code>eva02_S_pt_in21k_p14</code> on <b>IN-1K</b> using 1 nodes x 8 gpus per node (click to expand).</summary>

```bash   
MODEL_NAME=eva02_small_patch14_xattn_fusedLN_SwiGLU_preln_RoPE

PRETRAIN_CKPT=/path/to/eva02_S_pt_in21k_p14.pt 

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-1K


sz=336
batch_size=128  # 128(bsz_per_gpu)*8(#gpus_per_node)*1(#nodes)*1(update_freq)=1024(total_bsz)
update_freq=1

lr=1e-4           
lrd=0.8          

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=100
wmep=5
dpr=0.1

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.1

zero_stage=0

scale_low=0.08
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
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
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>






<details>
<summary>Fine-tune MIM pre-trained <code>eva02_B_pt_in21k_p14</code> on <b>IN-1K</b> using 4 nodes x 8 gpus per node (click to expand).</summary>

```bash   
MODEL_NAME=eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

PRETRAIN_CKPT=/path/to/eva02_B_pt_in21k_p14.pt 

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-1K


sz=448
batch_size=32  # 32(bsz_per_gpu)*8(#gpus_per_node)*4(#nodes)*1(update_freq)=1024(total_bsz)
update_freq=1

lr=1e-4           
lrd=0.7          

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=30
wmep=3
dpr=0.1

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.1

zero_stage=0

scale_low=0.08
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
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
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>







<details>
<summary>Fine-tune MIM pre-trained <code>eva02_L_pt_in21k_p14</code> on <b>IN-1K</b> using 4 nodes x 8 gpus per node (click to expand).</summary>

```bash   
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

PRETRAIN_CKPT=/path/to/eva02_L_pt_in21k_p14.pt 

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-1K


sz=448
batch_size=16   # 16(bsz_per_gpu)*8(#gpus_per_node)*4(#nodes)*2(update_freq)=1024(total_bsz)
update_freq=2

lr=5e-5        
lrd=0.8          

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=30
wmep=3
dpr=0.15

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.2

zero_stage=1

scale_low=0.08
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
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
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>







<details>
<summary>Fine-tune MIM pre-trained <code>eva02_L_pt_m38m_p14</code> on <b>IN-1K</b> using 4 nodes x 8 gpus per node (click to expand).</summary>

```bash   
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

PRETRAIN_CKPT=/path/to/eva02_L_pt_m38m_p14.pt 

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-1K


sz=448
batch_size=16   # 16(bsz_per_gpu)*8(#gpus_per_node)*4(#nodes)*2(update_freq)=1024(total_bsz)
update_freq=2

lr=7e-5        
lrd=0.8          

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=30
wmep=3
dpr=0.15

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.2

zero_stage=1

scale_low=0.08
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
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
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>















### Intermediate Fine-tune MIM pre-trained EVA-02 on IN-21K

<details>
<summary>Fine-tune MIM pre-trained <code>eva02_B_pt_in21k_p14</code> on <b>IN-21K</b> using 4 nodes x 8 gpus per node (click to expand).</summary>

```bash
MODEL_NAME=eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

PRETRAIN_CKPT=/path/to/eva02_B_pt_in21k_p14.pt

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-21K

sz=448
batch_size=64   # 64(bsz_per_gpu)*8(#gpus_per_node)*4(#nodes)*1(update_freq)=2048(total_bsz)
update_freq=1

lr=3e-4            
lrd=0.7          

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=40
wmep=1
dpr=0.1

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.1

zero_stage=1

scale_low=0.2
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=23333 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH} \
        --disable_eval_during_finetuning \
        --nb_classes 21841 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
        --input_size ${sz} \
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
        --scale ${scale_low} 1.0 \
        --aa ${aa} \
        --enable_deepspeed

```

</details>





<details>
<summary>Fine-tune MIM pre-trained <code>eva02_L_pt_in21k_p14</code> on <b>IN-21K</b> using 8 nodes x 8 gpus per node (click to expand).</summary>

```bash
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

PRETRAIN_CKPT=/path/to/eva02_L_pt_in21k_p14.pt

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-21K

sz=448
batch_size=16   # 16(bsz_per_gpu)*8(#gpus_per_node)*8(#nodes)*1(update_freq)=1024(total_bsz)
update_freq=1

lr=2e-4            
lrd=0.75          

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=20
wmep=1
dpr=0.15

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.1

zero_stage=1

scale_low=0.2
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=23333 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH} \
        --disable_eval_during_finetuning \
        --nb_classes 21841 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
        --input_size ${sz} \
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
        --scale ${scale_low} 1.0 \
        --aa ${aa} \
        --enable_deepspeed

```

</details>





<details>
<summary>Fine-tune MIM pre-trained <code>eva02_L_pt_m38m_p14</code> on <b>IN-21K</b> using 8 nodes x 8 gpus per node (click to expand).</summary>

```bash
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

PRETRAIN_CKPT=/path/to/eva02_L_pt_m38m_p14.pt

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-21K

sz=448
batch_size=16   # 16(bsz_per_gpu)*8(#gpus_per_node)*8(#nodes)*2(update_freq)=2048(total_bsz)
update_freq=2

lr=3e-4            
lrd=0.75          

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=30
wmep=1
dpr=0.15

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.1

zero_stage=1

scale_low=0.2
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=23333 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH} \
        --disable_eval_during_finetuning \
        --nb_classes 21841 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
        --input_size ${sz} \
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
        --scale ${scale_low} 1.0 \
        --aa ${aa} \
        --enable_deepspeed

```

</details>











### Fine-tune EVA-02 on IN-1K with IN-21K intermediate fine-tuned checkpoint

<details>
<summary>Fine-tune IN-21K-tuned <code>eva02_B_pt_in21k_medft_in21k_p14</code> on <b>IN-1K</b> using 1 nodes x 8 gpus per node (click to expand).</summary>

```bash   
MODEL_NAME=eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

PRETRAIN_CKPT=/path/to/eva02_B_pt_in21k_medft_in21k_p14.pt

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-1K


sz=448
batch_size=64   # 64(bsz_per_gpu)*8(#gpus_per_node)*1(#nodes)*1(update_freq)=512(total_bsz)
update_freq=1

lr=5e-5           
lrd=0.8           

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=15
wmep=2
dpr=0.15

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.2

zero_stage=1

scale_low=0.08
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1



python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
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
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>






<details>
<summary>Fine-tune IN-21K-tuned <code>eva02_L_pt_in21k_medft_in21k_p14</code> on <b>IN-1K</b> using 4 nodes x 8 gpus per node (click to expand).</summary>

```bash   
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

PRETRAIN_CKPT=/path/to/eva02_L_pt_in21k_medft_in21k_p14.pt

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-1K


sz=448
batch_size=16   # 16(bsz_per_gpu)*8(#gpus_per_node)*4(#nodes)*1(update_freq)=512(total_bsz)
update_freq=1

lr=2e-5           
lrd=0.85           

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=20
wmep=2
dpr=0.15

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.2

zero_stage=1

scale_low=0.08
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1



python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
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
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>






<details>
<summary>Fine-tune IN-21K-tuned <code>eva02_L_pt_m38m_medft_in21k_p14</code> on <b>IN-1K</b> using 4 nodes x 8 gpus per node (click to expand).</summary>

```bash   
MODEL_NAME=eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE

PRETRAIN_CKPT=/path/to/eva02_L_pt_m38m_medft_in21k_p14.pt

OUTPUT_DIR=/path/to/output/{MODEL_NAME}

DATA_PATH=/path/to/IN-1K


sz=448
batch_size=16   # 16(bsz_per_gpu)*8(#gpus_per_node)*4(#nodes)*1(update_freq)=512(total_bsz)
update_freq=1

lr=2e-5           
lrd=0.85           

warmup_lr=0.0
min_lr=0.0
weight_decay=0.05

partial_freeze=0
ep=20
wmep=2
dpr=0.15

reprob=0.0
mixup=0.0
cutmix=0.0
smoothing=0.2

zero_stage=1

scale_low=0.08
crop_pct=1.0
aa=rand-m9-mstd0.5-inc1



python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${WORLD_SIZE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=12345 --use_env run_class_finetuning.py \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir ${OUTPUT_DIR} \
        --log_dir ${OUTPUT_DIR}/tb_log \
        --model ${MODEL_NAME} \
        --finetune ${PRETRAIN_CKPT} \
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
        --model_ema \
        --model_ema_eval \
        --enable_deepspeed
```

</details>





## Acknowledgement

EVA-02 is built using the awesome [EVA-01](https://github.com/baaivision/EVA/tree/master/EVA-01), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [BEiTv2](https://github.com/microsoft/unilm/tree/master/beit), [CLIP](https://github.com/openai/CLIP), [MAE](https://github.com/facebookresearch/mae/), [timm](https://github.com/rwightman/pytorch-image-models), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [Apex](https://github.com/NVIDIA/apex), [xFormer](https://github.com/facebookresearch/xformers), and [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch).
