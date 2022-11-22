# Contrastive Language-Image Pre-Training with EVA (EVA-CLIP)

**Table of Contents**

- [Contrastive Language-Image Pre-Training with EVA (EVA-CLIP)](#contrastive-language-image-pre-training-with-eva-eva-clip)
  - [Model Card](#model-card)
  - [Usage](#usage)
  - [Acknowledgement](#acknowledgement)
  

## Model Card

<div align="center">

| model name | #param. | precision | data  |  batch size | IN-1K zero-shot top-1 | weight |
|:-----------:|:------:|:------:|:------:|:------:|:------:|:------:|
| `eva_clip_psz14` | 1.3B | `fp16` | [LAION-400M](https://laion.ai/laion-400-open-dataset/) | 41K | 78.5 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) (`2GB`) |

</div>

> The ImageNet-1K zero-shot classification performance is higher than our paper (`78.5` *v.s.* `78.2`) because of longer training.

We choose to train a 1.3B CLIP model, not because it is easy, but because it is hard. Please refer to [this note](https://docs.google.com/document/d/1FXosAZ3wMrzThgnWR6KSkXIz4IMItq3umDGos38pJps/edit) for a glance of the challenges in training very large CLIP.

To our knowledge, EVA-CLIP is **the largest performant open-sourced CLIP model** evaluated via zero-shot classification performance.
We will updates the results in our paper soon.
For more details of EVA-CLIP, please refer to Section 2.3.5 of [our paper](https://arxiv.org/pdf/2211.07636.pdf).

We hope open-sourcing EVA-CLIP can facilitate future research in multi-modal learning, representation leaning, AIGC, *etc*.


## Usage

The usege of EVA-CLIP is similar to [OpenAI CLIP](https://github.com/openai/CLIP) and [Open CLIP](https://github.com/mlfoundations/open_clip).
Here we provide a showcase in zero-shot image classification.

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
```

The training code of our 1.3B EVA-CLIP will be available at [FlagAI](https://github.com/FlagAI-Open/FlagAI). Please stay tuned.


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
EVA-CLIP is bulit with [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip) and [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark). Thanks for their awesome work!
