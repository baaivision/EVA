# EVA: Video Action Recognition

**Table of Contents**

- [EVA: Video Action Recognition](#eva-video-action-recognition)
  - [Model Card](#model-card)
    - [Prepare EVA pre-trained weight](#prepare-eva-pre-trained-weight)
    - [Kinetics fine-tuned weights](#kinetics-fine-tuned-weights)
  - [Setup](#setup)
  - [Datasets](#datasets)
    - [Prepare videos](#prepare-videos)
    - [Generate file list](#generate-file-list)
  - [Evaluation](#evaluation)
    - [Kinetics-400 Evaluation](#kinetics-400-evaluation)
    - [Kinetics-600 Evaluation](#kinetics-600-evaluation)
    - [Kinetics-700 Evaluation](#kinetics-700-evaluation)
  - [Training](#training)
    - [Kinetics-722 intermediate fine-tune](#kinetics-722-intermediate-fine-tune)
    - [Kinetics-400 fine-tune](#kinetics-400-fine-tune)
    - [Kinetics-600 fine-tune](#kinetics-600-fine-tune)
    - [Kinetics-700 fine-tune](#kinetics-700-fine-tune)
  - [Acknowledgment](#acknowledgment)

## Model Card
We provide all checkpoints of our EVAs for video recognition.

### Prepare EVA pre-trained weight

<div align="center">

| model name | #param. |pre-training epochs on merged-30M |                                      weight                                      |
|------------|:------:|:------------------:|:--------------------------------------------------------------------------------:|
| `eva_psz14` | 1.0B | 150 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14.pt) (`2GB`) |

</div>

EVA is an open billion-scale vision foundation model, [pre-trained](../eva) on the merged-30M dataset.


### Kinetics fine-tuned weights

<div align="center">

|   dataset   |     model name     |                                     init. weight                                     |  acc@1   |                       config                       |                                          weight                                          |                   logs                   |
|:-----------:|:------------------:|:------------------------------------------------------------------------------------:|:--------:|:--------------------------------------------------:|:----------------------------------------------------------------------------------------:|:----------------------------------------:|
| Kinetics722 |  `eva_video_k722`  |      [`eva_psz14`](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14.pt)       |    -     | [config](configs/kinetics722_intermediate_ft.yaml) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k722.pth) (`4.8GB`) | [ft_k722](../logs/video/ft_k722_log.txt) |
| Kinetics400 |  `eva_video_k400`  | [`eva_video_k722`](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k722.pth) | **89.7** |       [config](configs/kinetics400_ft.yaml)        | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k400.pth) (`4.8GB`) | [ft_k400](../logs/video/ft_k400_log.txt) |
| Kinetics600 |  `eva_video_k600`  | [`eva_video_k722`](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k722.pth) | **89.8** |       [config](configs/kinetics600_ft.yaml)        | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k600.pth) (`4.8GB`) | [ft_k600](../logs/video/ft_k600_log.txt) |                                                         
| Kinetics700 |  `eva_video_k700`  | [`eva_video_k722`](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k722.pth) | **82.9** |       [config](configs/kinetics700_ft.yaml)        | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k700.pth) (`4.8GB`) | [ft_k700](../logs/video/ft_k700_log.txt) |                                                      

</div>

All pre-trained weights can be downloaded using the following script. 
If problems occur with the automatic download, please follow the instructions for a manual download within the script.
```bash
sh scripts/download_checkpoints.sh
```


## Setup

To set up the environment, run the following:
```bash
conda create -n evavideo python=3.7
conda activate evavideo
pip install -r requirements.txt
```

Build torch w.r.t. cuda version from conda
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

Install [Apex](https://github.com/NVIDIA/apex) as follows
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


## Datasets
We have successfully fine-tuned our EVA on our merged `Kinetics-722` and `Kinetics-400/600/700`  with this codebase.
To make video decoding faster, we use [decord](https://github.com/zhreshold/decord) to decode the videos on the fly.

### Prepare videos
Please refer to the [official website](https://deepmind.com/research/open-source/kinetics) and/or the official script to prepare the videos. 


Symlink the downloaded dataset
```bash
ln -s /path_to_Kinectics-400_dataset data/k400
ln -s /path_to_Kinectics-600_dataset data/k600
ln -s /path_to_Kinectics-700_dataset data/k700
```

The folder structure should look like this:
```bash
video
├── ...
├── data
│   ├── k400/600/700  -> path_to_Kinetics-400/600/700
│   │   ├── train
│   │   │   ├── ${CLASS_NAME}/${VIDEO_ID}
│   │   ├── val
│   │   │   ├── ${CLASS_NAME}/${VIDEO_ID}
│   ├── k400/600/700/722_train.txt
│   ├── k400/600/700/722_val.txt
│   ├── k722_to_k400/600/700_mapping.npy
├── ...
```

### Generate file list
We provide a convenient script to generate an annotation file list. Please follow [`notebooks/build_file_list.ipynb`](notebooks/build_file_list.ipynb) to generate file lists given downloaded videos.


The merged dataset coined Kinetics-722 (K-722) integrates all valid training samples from Kinetics-400 (K-400), Kinetics-600 (K-600), and Kinetics-700 (K-700). 
Notably, for a fair and legal comparison, we removed leaked videos in all validation sets and duplicated videos in all training sets based on `youtube id` of the video.  
Accordingly, the cleaned K-722 contains 0.63M training videos, covering 722 human action classes. We also provide [our data list]( https://huggingface.co/BAAI/EVA/blob/main/eva%20video%20data%20list.zip).

Now, you can train and test EVA on video data.

**Note:**
Since our method is built upon [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP), it needs a textual description for each video category. For example, we provide the text description of `Kinetics-722` in the file [`labels/kinetics722_labels.csv`](labels/kinetics722_labels.csv). Here is the format:
```bash
$ head -n 5 labels/kinetics722_labels.csv
id,name
0,abseiling
1,acting in play
2,adjusting glasses
3,air drumming
```
The `id` indicates the class id, while the `name` denotes the text description.
Note that we disabled the text branch.  



## Evaluation

### Kinetics-400 Evaluation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=eva-exploring-the-limits-of-masked-visual)

To evaluate EVA with 16 frames on **Kinetics-400** using a single node with 8 gpus:
- multi-view evaluation
```bash
VIDEO_CONFIG=configs/kinetics400_ft.yaml

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 main.py -cfg ${SEG_CONFIG} --only_test --resume pretrained/eva_video_k400.pth \
--output /path/to/output --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3

# expected results
# top-1 accuracy: 89.7
```


### Kinetics-600 Evaluation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/action-classification-on-kinetics-600)](https://paperswithcode.com/sota/action-classification-on-kinetics-600?p=eva-exploring-the-limits-of-masked-visual)

To evaluate EVA with 16 frames on **Kinetics-600** using a single node with 8 gpus:
- multi-view evaluation
```bash
VIDEO_CONFIG=configs/kinetics600_ft.yaml

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 main.py -cfg ${SEG_CONFIG} --only_test --resume pretrained/eva_video_k600.pth \
--output /path/to/output --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3

# expected results
# top-1 accuracy: 89.8
```


### Kinetics-700 Evaluation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/action-classification-on-kinetics-700)](https://paperswithcode.com/sota/action-classification-on-kinetics-700?p=eva-exploring-the-limits-of-masked-visual)

To evaluate EVA with 16 frames on **Kinetics-700** using a single node with 8 gpus:
- multi-view evaluation
```bash
VIDEO_CONFIG=configs/kinetics700_ft.yaml

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NNODES --node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR --master_port=12355 main.py -cfg ${SEG_CONFIG} --only_test --resume pretrained/eva_video_k700.pth \
--output /path/to/output --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3

# expected results
# top-1 accuracy: 82.9
```

## Training
The config files lie in [`configs`](configs). 

### Kinetics-722 intermediate fine-tune
To train EVA with 8 frames on **Kinetics-722** using 16 nodes (`total_batch_size=256`):
```bash
VIDEO_CONFIG=configs/kinetics722_intermediate_ft.yaml
OUTPUT_ROOT=/path/to/video/output/
pretrained=pretrained/eva_psz14.pt # https://huggingface.co/BAAI/EVA/blob/main/eva_psz14.pt
    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes \
--node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 \
main.py -cfg ${VIDEO_CONFIG} \
--output ${OUTPUT_ROOT} \
--accumulation-steps 1 \
--opts MODEL.PRETRAINED ${pretrained}
```

### Kinetics-400 fine-tune
For example, to train EVA with 16 frames on **Kinetics-400** using 8 nodes (`total_batch_size=256`):
```bash
VIDEO_CONFIG=configs/kinetics400_ft.yaml
OUTPUT_ROOT=/path/to/video/output/
pretrained=pretrained/eva_video_k722.pth # https://huggingface.co/BAAI/EVA/blob/main/eva_video_k722.pth
    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes \
--node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 \
main.py -cfg ${VIDEO_CONFIG} \
--output ${OUTPUT_ROOT} \
--accumulation-steps 4 \
--opts MODEL.PRETRAINED ${pretrained}
```

### Kinetics-600 fine-tune
For example, to train EVA with 16 frames on **Kinetics-600** using 8 nodes (`total_batch_size=256`):
```bash
VIDEO_CONFIG=configs/kinetics600_ft.yaml
OUTPUT_ROOT=/path/to/video/output/
pretrained=pretrained/eva_video_k722.pth
    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes \
--node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 \
main.py -cfg ${VIDEO_CONFIG} \
--output ${OUTPUT_ROOT} \
--accumulation-steps 4 \
--opts MODEL.PRETRAINED ${pretrained}
```


### Kinetics-700 fine-tune
For example, to train EVA with 16 frames on **Kinetics-700** using 8 nodes (`total_batch_size=256`):
```bash
VIDEO_CONFIG=configs/kinetics700_ft.yaml
OUTPUT_ROOT=/path/to/video/output/
pretrained=pretrained/eva_video_k722.pth
    
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$nnodes \
--node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12355 \
main.py -cfg ${VIDEO_CONFIG} \
--output ${OUTPUT_ROOT} \
--accumulation-steps 4 \
--opts MODEL.PRETRAINED ${pretrained}
```



**Note:**
- We recommend setting the total batch size to 256. If memory or #GPUs is limited, you can use `--accumulation-steps` to maintain the total batch size. Specifically, here the effective total batch size is 64(`GPUs_NUM`) x 1(`TRAIN.BATCH_SIZE`) x 4(`TRAIN.ACCUMULATION_STEPS`) = 256.
- Please specify the data path in config file(`configs/*.yaml`). Also, you can set them by attaching an argument `--opts DATA.ROOT /path/to/data DATA.TRAIN_FILE /path/to/train/list DATA.VAL_FILE /path/to/val/list`.

*Disclaimer:*
- Due to differences in Kinetics datasets, one of the uncertainties now is the generation of the data list in the `notebooks`.

## Acknowledgment
EVA video action recognition is build with [mmaction2](https://github.com/open-mmlab/mmaction2), [Swin](https://github.com/microsoft/Swin-Transformer), [CLIP](https://github.com/openai/CLIP) and [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP). 
Thanks for their wonderful work!