---
license: mit
---

<div align="center">
<h1>EVA: An Open Billion-Scale Vision Foundation Model </h1>
<h3><a href="https://arxiv.org/abs/2211.07636">EVA: Exploring the Limits of Masked Visual Representation Learning at Scale</a></h3>

[Yuxin Fang](https://bit.ly/YuxinFang_GoogleScholar)<sup>2,1</sup>, [Wen Wang](https://scholar.google.com/citations?user=1ks0R04AAAAJ&hl)<sup>3,1</sup>, [Binhui Xie](https://binhuixie.github.io/)<sup>4,1</sup>, [Quan Sun](https://github.com/Quan-Sun)<sup>1</sup>, [Ledell Wu](https://scholar.google.com/citations?user=-eJHVt8AAAAJ&hl=en)<sup>1</sup>, [Xinggang Wang](https://xinggangw.info/)<sup>2</sup>, [Tiejun Huang](https://scholar.google.com/citations?user=knvEK4AAAAAJ&hl=en)<sup>1</sup>, [Xinlong Wang](https://www.xloong.wang/)<sup>1</sup>, [Yue Cao](http://yue-cao.me/)<sup>1</sup>
 
<sup>1</sup>[BAAI](https://www.baai.ac.cn/english.html), <sup>2</sup>[HUST](http://english.hust.edu.cn/), <sup>3</sup>[ZJU](https://www.zju.edu.cn/english/), <sup>4</sup>[BIT](https://english.bit.edu.cn/)


We launch **EVA**, a vision-centric foundation model to **E**xplore the limits of **V**isual representation at sc**A**le using only publicly accessible data and academic resources. **EVA** is a vanilla ViT pre-trained to reconstruct the masked out image-text aligned vision features (*i.e.*, CLIP features) conditioned on visible image patches. Via this pretext task, we can efficiently scale up EVA to one billion parameters, and sets new records on a broad range of representative vision downstream tasks.

***EVA is the first open-sourced billion-scale vision foundation model that achieves state-of-the-art performance on a broad range of downstream tasks.***

</div>


**Table of Contents**
- [Image Classification](#image-classification)
  - [Summary of EVA's image classification performance](#summary-of-evas-image-classification-performance)
- [Video Classification](#video-classification)
- [Object Detection \& Instance Segmentation](#object-detection--instance-segmentation)
  - [COCO 2017 (single-scale evaluation on `val` set)](#coco-2017-single-scale-evaluation-on-val-set)
  - [LVIS v1.0 (single-scale evaluation on `val` set)](#lvis-v10-single-scale-evaluation-on-val-set)
- [Semantic Segmentation](#semantic-segmentation)
  - [COCO-Stuff-164K](#coco-stuff-164k)
  - [ADE20K](#ade20k)
- [EVA-CLIP](#eva-clip)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
   

## Image Classification 

We provide **all pre-trained & fine-tuned** EVAs for the community. 
The following table summarizes the basic statistics of MIM pre-trained EVA and image classification EVA.

| model name | #param. |pre-training epochs on merged-30M | intermeidate fine-tuning epochs on IN-21K | fine-tuning epochs on IN-1K | IN-1K top-1 acc. |weight |
|------------|:------:|:------------------:|:------:|:------:|:------:|:------:|
| `eva_psz14` | 1.0B | 150 | - | - | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14.pt) (`2GB`) |
| `eva_psz14to16` | 1.0B | 150 | - | - | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt) (`2GB`) | 
| `eva_21k_224px_psz14` | 1.0B | 150 | 60 | - | - | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_21k_224px_psz14.pt) (`2GB`) |
| `eva_21k_1k_336px_psz14_ema` | 1.0B | 150 | 60 | 10 | **89.6** | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_336px_psz14_ema_89p6.pt) (`4GB`) |
| `eva_21k_1k_560px_psz14_ema` | 1.0B | 150 | 60 | 15 | **89.7** | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_21k_1k_560px_psz14_ema_89p7.pt) (`4GB`) |

- `eva_psz14to16` model interpolates the kernel size of `patch_embed` from `14x14` to `16x16`. This is useful for object detection, instance segmentation & semantic segmentation, *etc*. See [`interpolate_patch_14to16.py`](interpolate_patch_14to16.py) for implementation details.
- For MIM pre-trained EVA and EVA-CLIP, we use `deepspeed` `fp16` format. IN-1K fine-tuned EVA weights are larger (`4GB` *v.s.* `2GB`) because ema updates models with `fp32` format. The weights of other downstream tasks are also with `fp32` format.

</div>


### Summary of EVA's image classification performance

<div align="center">

| model | [IN-1K](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet.csv) | [IN-V2](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv) | [IN-ReaL](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-real.csv) | [IN-Adv.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-a.csv) | [IN-Ren.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv) | [IN-Ske.](https://github.com/rwightman/pytorch-image-models/blob/main/results/results-imagenet-r.csv) | ObjectNet |
|:------------:|:------------------:|:------:|:------:| :------:|:------:|:------:|:------:|
| EVA | 89.6 | 81.6 | 90.8 | 86.2 | 88.3 | 67.7 | 60.9 |

</div>


## Video Classification

<div align="center">

|   dataset   |     model name     |                                     init. weight                                     |  acc@1   |                       config                       |                                          weight                                          |                   logs                   |
|:-----------:|:------------------:|:------------------------------------------------------------------------------------:|:--------:|:--------------------------------------------------:|:----------------------------------------------------------------------------------------:|:----------------------------------------:|
| Kinetics722 |  `eva_video_k722`  |      [`eva_psz14`](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14.pt)       |    -     | [config](configs/kinetics722_intermediate_ft.yaml) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k722.pth) (`4.8GB`) | [ft_k722](../logs/video/ft_k722_log.txt) |
| Kinetics400 |  `eva_video_k400`  | [`eva_video_k722`](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k722.pth) | **89.7** |       [config](configs/kinetics400_ft.yaml)        | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k400.pth) (`4.8GB`) | [ft_k400](../logs/video/ft_k400_log.txt) |
| Kinetics600 |  `eva_video_k600`  | [`eva_video_k722`](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k722.pth) | **89.8** |       [config](configs/kinetics600_ft.yaml)        | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k600.pth) (`4.8GB`) | [ft_k600](../logs/video/ft_k600_log.txt) |                                                         
| Kinetics700 |  `eva_video_k700`  | [`eva_video_k722`](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k722.pth) | **82.9** |       [config](configs/kinetics700_ft.yaml)        | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_video_k700.pth) (`4.8GB`) | [ft_k700](../logs/video/ft_k700_log.txt) |                                                      

</div>


## Object Detection & Instance Segmentation

<div align="center">

| model name | #param. | pre-training interations on Objects365 |                                    weight                                     |
|------------|:-------:|:--------------------------------------:|:-----------------------------------------------------------------------------:|
| `eva_o365` |  1.1B   |                  380k                  |       [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_o365.pth) (`4GB`)        |

</div>


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


## Semantic Segmentation

### COCO-Stuff-164K

<div align="center">

| init. model weight | batch size | iter | crop size | mIoU (ss) | config | seg model weight |logs|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`eva_psz14to16`](https://huggingface.co/BAAI/EVA/blob/main/eva_psz14to16.pt) | 32 | 60k | 896 | **53.4** | [config](configs/coco_stuff164k/eva_mask2former_896_60k_cocostuff164k_ss.py) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_53p4.pth) | [training](../logs/sem_seg/ft_cocstuff164k_sem_seg_ss_53p4_training_log.txt) \| [evaluation](../logs/sem_seg/ft_cocstuff164k_sem_seg_ss_53p4.txt)

</div>

### ADE20K

<div align="center">

| init. model weight | batch size | iter | crop size | mIoU | config | seg model weight |logs|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`eva_sem_seg_coco`](https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_cocostuff_53p4.pth) | 64 | 20k | 896 | **61.5** (ss) \| **62.3** (ms) | [config](configs/ade20k/eva_mask2former_896_20k_coco164k2ade20k_ss.py) | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_sem_seg_mask2former_ade_ss61p5_ms62p3.pth) | [training](../logs/sem_seg/ft_ade20k_sem_seg_ms_62p3_training_log.txt) \| [evaluation](../logs/sem_seg/ft_ade20k_sem_seg_ms_62p3.txt)

</div>



## EVA-CLIP


<div align="center">

| model name | #param. | precision | data  |  batch size | IN-1K zero-shot top-1 | weight |
|:-----------:|:------:|:------:|:------:|:------:|:------:|:------:|
| `eva_clip_psz14` | 1.3B | `fp16` | [LAION-400M](https://laion.ai/laion-400-open-dataset/) | 41K | **78.5** | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) (`2GB`) |

</div>

> The ImageNet-1K zero-shot classification performance is higher than our paper (`78.5` *v.s.* `78.2`) because of longer training.

We choose to train a 1.3B CLIP model, not because it is easy, but because it is hard. Please refer to [this note](https://docs.google.com/document/d/1FXosAZ3wMrzThgnWR6KSkXIz4IMItq3umDGos38pJps/edit) for a glance of the challenges in training very large CLIP.

To our knowledge, EVA-CLIP is **the largest performant open-sourced CLIP model** evaluated via zero-shot classification performance.
We will updates the results in our paper soon.
For more details of EVA-CLIP, please refer to Section 2.3.5 of [our paper](https://arxiv.org/pdf/2211.07636.pdf).

We hope open-sourcing EVA-CLIP can facilitate future research in multi-modal learning, representation leaning, AIGC, *etc*.



## Citation
If you find our work helpful, please star this repo and cite the related articles. Thanks for your support!

```
@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}
```


## License

The content of this project itself is licensed under the MIT License.

## Contact

For help or issues using EVA, please open a GitHub [issue](https://github.com/baaivision/EVA/issues/new).

**We are hiring** at all levels at BAAI Vision Team, including full-time researchers, engineers and interns. 
If you are interested in working with us on **foundation model, self-supervised learning and multimodal learning**, please contact [Yue Cao](http://yue-cao.me/) (`caoyue@baai.ac.cn`) and [Xinlong Wang](https://www.xloong.wang/) (`wangxinlong@baai.ac.cn`).
