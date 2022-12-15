# Contrastive Language-Image Pre-Training with EVA (EVA-CLIP)

**Table of Contents**

- [Contrastive Language-Image Pre-Training with EVA (EVA-CLIP)](#contrastive-language-image-pre-training-with-eva-eva-clip)
  - [Model Card](#model-card)
  - [Performance of EVA-CLIP Vision Encoder on ImageNet-1K](#performance-of-eva-clip-vision-encoder-on-imagenet-1k)
  - [EVA-CLIP Zero-shot Evaluation Results](#eva-clip-zero-shot-evaluation-results)
    - [**All 35 Benchmark Results in Details**](#all-35-benchmark-results-in-details)
    - [Zero-shot Image Classification Evaluation](#zero-shot-image-classification-evaluation)
    - [Zero-shot Video Action Recognition Evaluation](#zero-shot-video-action-recognition-evaluation)
    - [Zero-shot Retrieval Evaluation](#zero-shot-retrieval-evaluation)
  - [Usage](#usage)
  - [Acknowledgement](#acknowledgement)
  

## Model Card

<div align="center">

| model name | #param. | precision | data  |  batch size | IN-1K zero-shot top-1 | weight |
|:-----------:|:------:|:------:|:------:|:------:|:------:|:------:|
| `eva_clip_psz14` | 1.1B | `fp16` | [LAION-400M](https://laion.ai/laion-400-open-dataset/) | 41K | 78.5 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) (`2GB`) |

</div>


We choose to train a 1.1B CLIP model, not because it is easy, but because it is hard. Please refer to [this note](https://docs.google.com/document/d/1FXosAZ3wMrzThgnWR6KSkXIz4IMItq3umDGos38pJps/edit) for a glance at the challenges in training very large CLIP.

To our knowledge, EVA-CLIP is **the largest performant open-sourced CLIP model** evaluated via zero-shot classification performance, especially on mainstream classification benchmarks such as ImageNet along with its variants. 
For more details about EVA-CLIP, please refer to Section 2.3.5 of [our paper](https://arxiv.org/pdf/2211.07636.pdf).

We hope open-sourcing EVA-CLIP can facilitate future research in multi-modal learning, representation learning, AIGC, *etc*, and we hope our solution for scaling up CLIPs can provide insight for practitioners studying large foundation models.



## Performance of EVA-CLIP Vision Encoder on ImageNet-1K

<div align="center">

| model | zero-shot @ 224px | linear probing @ 224px | linear probing @ 336px | fine-tuning @ 224px | fine-tuning @ 336px |
|:-----:|:------:|:------:|:------:|:------:|:------:| 
| EVA-CLIP | **78.5** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) \| [log](https://wandb.ai/baaivision/eva-clip/reports/ViT-g-14--VmlldzoyOTkwMDYy)) | **86.5** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz224_lincls_86p5.pth) \| [log](../logs/cls/linear_eva_clip_vision_enc_1k_cls_sz224_86p5.txt)） | **86.5** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz336_lincls_86p5.pth) \| [log](../logs/cls/linear_eva_clip_vision_enc_1k_cls_sz336_86p5.txt)) | **89.1** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz224_ftcls_89p1.pt) \| [log](../logs/cls/ft_eva_clip_vision_enc_1k_cls_sz224_89p1.txt)) | **89.4** ([weight](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_vis_enc_sz336_ftcls_89p4.pt) \| [log](../logs/cls/ft_eva_clip_vision_enc_1k_cls_sz336_89p4.txt)) |

</div>

EVA-CLIP achieves the state-of-the-art top-1 accuracy on ImageNet-1K among all self-supervised learning approaches.
We will provide instructions for re-producing these results soon.


## EVA-CLIP Zero-shot Evaluation Results


<div align="center">

### [**All 35 Benchmark Results in Details**](./benchmark.md) 

</div>



### Zero-shot Image Classification Evaluation

The top-1 accuracy of ImageNet-1K variants and ObjectNet.

<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/self-supervised-image-classification-with)](https://paperswithcode.com/sota/self-supervised-image-classification-with?p=eva-exploring-the-limits-of-masked-visual) 

| model | IN-1K | IN-V2 |  IN-Adv. | IN-Ren. |IN-Ske. | ObjectNet |
|-------|:-----:|:-----:|:----:| :------:|:-------:|:---------:|
| OpenAI CLIP-L | 75.55 | 69.86 | 70.76 | 87.83 | 59.58 | 68.98 |
| Open CLIP-H | 77.96 | 70.87 | 59.33 | 89.33 | 66.58 | 69.71 |
| Open CLIP-g | 76.65 | 69.56 | 57.19 | 88.69 | 65.17 | 67.53 |
| EVA CLIP-g | **78.53** | **71.52** | **73.59** | **92.5** | **67.31** | **72.33** |
 
</div>

### Zero-shot Video Action Recognition Evaluation

The performance of video action recognition benchmarks.

<div align="center">

| model | UCF-101 | Kinetics-400 | Kinetics-600 | Kinetics-700 |
|-------|:-----:|:-----:|:----:| :----:|
| OpenAI CLIP-L | 76.39 | 64.47 | 64.21 | 57.68 |
| Open CLIP-H   | **78.16** | 63.06 | 63.58 | 56.09 |
| Open CLIP-g   | 77.73 | 61.69 | 62.16 | 54.99 |
| EVA CLIP-g    | 76.05 | **65.23** | **64.38** | **58.4** |

</div>


> For video action recognition, we sample only a single center frame each video, turning it into an image classification task.
> Following the conventional settings, we report the top-1 accuracy for UCF-101 and the mean of top-1 and top-5 accuracy for Kinetics-400/600/700.

### Zero-shot Retrieval Evaluation

<div align="center">

<table>
   <tr>
      <td rowspan=2>Dataset</td>
      <td rowspan=2>Model</td>
      <td colspan=3>Text-to-Image Retrival</td>
      <td colspan=3>Image-to-Text Retrival</td>
   </tr>
   <tr>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
   </tr>
   <tr>
      <td rowspan=4>Flickr30k</td>
      <td>OpenAI CLIP-L</td>
      <td>65.18 </td>
      <td>87.28 </td>
      <td>92 </td>
      <td>85.2 </td>
      <td>97.3 </td>
      <td>99 </td>
   </tr>
   <tr>
      <td>Open CLIP-H</td>
      <td><b>77.78</b></td>
      <td><b>94.14</b></td>
      <td><b>96.62</b></td>
      <td><b>90.8</b></td>
      <td><b>99.3</b></td>
      <td>99.7</td>
   </tr>
   <tr>
      <td>Open CLIP-g</td>
      <td>76.52 </td>
      <td>93.62 </td>
      <td>96.28 </td>
      <td>90.8 </td>
      <td>99.1 </td>
      <td><b>99.8</b></td>
   </tr>
   <tr>
      <td>EVA CLIP-g</td>
      <td>72.64 </td>
      <td>91.6 </td>
      <td>95.12 </td>
      <td>88.3 </td>
      <td>98.3 </td>
      <td>99.3 </td>
   </tr>
   <tr>
      <td rowspan=4>MSCOCO</td>
      <td>OpenAI CLIP-L</td>
      <td>36.51 </td>
      <td>61.01 </td>
      <td>71.11 </td>
      <td>56.34 </td>
      <td>79.32 </td>
      <td>86.66 </td>
   </tr>
   <tr>
      <td>Open CLIP-H</td>
      <td><b>49.47</b></td>
      <td><b>73.4</b></td>
      <td><b>81.53</b></td>
      <td><b>65.96</b></td>
      <td><b>86.06</b></td>
      <td><b>91.9</b></td>
   </tr>
   <tr>
      <td>Open CLIP-g</td>
      <td>47.99 </td>
      <td>72.37 </td>
      <td>80.75 </td>
      <td>64.96 </td>
      <td>85.3 </td>
      <td>91.46 </td>
   </tr>
   <tr>
      <td>EVA CLIP-g</td>
      <td>44.07 </td>
      <td>68.5 </td>
      <td>77.33 </td>
      <td>61.76 </td>
      <td>83.28 </td>
      <td>89.96 </td>
   </tr>
</table>

</div>

> The zero-shot retrieval performance of EVA-CLIP is relatively inferior to the Open CLIP-H / -g counterpart. We speculate there are two main reasons: 
> - The size / capacity of the language tower in EVA-CLIP is much smaller / weaker than Open CLIP-H and Open CLIP-g, *i.e.*, `124M` *v.s.* `354M`, and is only `~1/8` of the vision tower. Meanwhile, retrieval tasks depend more on the capacity of the language branch compared with classification tasks.
> - Retrieval tasks seem benefit more from the training dataset size (LAION-2B used by Open CLIP), while we only leverage LAION-400M for EVA-CLIP training. 
> Nevertheless, it is hard to make a head-to-head comparison between different CLIP models. In the future, we will further scale up the language encoder & training data to improve the retrieval performance.

## Usage

The use of EVA-CLIP is similar to [OpenAI CLIP](https://github.com/openai/CLIP) and [Open CLIP](https://github.com/mlfoundations/open_clip).
Here we provide a showcase of zero-shot image classification.

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
```

The training code of our 1.1B EVA-CLIP will be available at [FlagAI](https://github.com/FlagAI-Open/FlagAI). Please stay tuned.


An example:
```python
import torch
from eva_clip import build_eva_model_and_transforms
from clip import tokenize
from PIL import Image

eva_clip_path = "/path/to/eva_clip_psz14.pt" # https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt
model_name = "EVA_CLIP_g_14"
image_path = "CLIP.png"
caption = ["a diagram", "a dog", "a cat"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = build_eva_model_and_transforms(model_name, pretrained=eva_clip_path)
model = model.to(device)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = tokenize(caption).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [1.0000e+00, 2.0857e-10, 4.8534e-12]
```


## Acknowledgement
EVA-CLIP is built with [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip) and [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark). Thanks for their awesome work!
