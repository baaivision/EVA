# Contrastive Language-Image Pre-Training with EVA (EVA-CLIP)

## Model Card

| model name | #param. | precision | data  |  batch size | IN-1K zero-shot top-1 | weight |
|:-----------:|:------:|:------:|:------:|:------:|:------:|:------:|
| `eva_clip_psz14` | 1.3B | `fp16` | [LAION-400M](https://laion.ai/laion-400-open-dataset/) | 41K | 78.5 | [🤗 HF link](https://huggingface.co/BAAI/EVA/blob/main/eva_clip_psz14.pt) (`2GB`) |

The ImageNet-1K zero-shot classification performance is higher than our paper (`78.5` *v.s.* `78.2`) because of longer training.

We choose to train a 1.3B CLIP model, not because it is easy, but because it is hard. Please refer to [this note](https://docs.google.com/document/d/1FXosAZ3wMrzThgnWR6KSkXIz4IMItq3umDGos38pJps/edit) for a glance of the challenges in training very large CLIP.

To our knowledge, EVA-CLIP is the largest performant open-sourced CLIP model evaluated via zero-shot classification performance.
We will updates the results in our paper soon.

For more details of EVA-CLIP, please refer to Section 2.3.5 of [our paper](https://arxiv.org/pdf/2211.07636.pdf).


## Usage

The usege of EVA-CLIP is similar to [OpenAI CLIP](https://github.com/openai/CLIP) and [Open CLIP](https://github.com/mlfoundations/open_clip).


```python
import torch
from clip import build_eva_model_and_transforms, tokenize
from PIL import Image

eva_clip_path = /path/to/eva_clip_psz14.pt
model_name = "EVA_CLIP_g_14"
image_path = /path/to/image.png
caption = ["a diagram", "a dog", "a cat"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = build_eva_model_and_transforms(model_name, pretrained=eva_clip_path)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = tokenize(caption).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```
## API of EVA-CLIP
The EVA_CLIP module `clip` provides the following methods:

`clip.build_eva_model_and_transforms(model_name, pretrained)`

Returns the model and the TorchVision transform needed by the model, specified by the model name EVA_CLIP_g_14. It will download the model as necessary. The pretrained argument can be a path to a local checkpoint.

`clip.tokenize(text: Union[str, List[str]], context_length=77)`

Returns a LongTensor containing tokenized sequences of given text input(s). This can be used as the input to the model

---------------------------------------------
The model returned by `clip.build_eva_model_and_transforms()` supports the following methods:

`model.encode_image(image: Tensor)`

Given a batch of images, returns the image features encoded by the vision portion of the EVA_CLIP model.

`model.encode_text(text: Tensor)`

Given a batch of text tokens, returns the text features encoded by the language portion of the EVA_CLIP model.

`model(image: Tensor, text: Tensor)`

Given a batch of images and a batch of text tokens, returns two Tensors, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features, times 100.

## Zero-Shot Prediction
The code below performs zero-shot prediction using EVA_CLIP. This example takes an image from the CIFAR-100 dataset, and predicts the most likely labels among the 100 textual labels from the dataset.

```python
import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
eva_clip_path = /path/to/eva_clip_psz14.pt

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.create_model_and_transforms('EVA_CLIP_g_14', eva_clip_path)

model = model.to(device)
model.eval()

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")

```
The output will look like the following (the exact numbers may be slightly different depending on the compute device):
```bash
Top predictions:

           snake: 100.00%
          turtle: 0.00%
     caterpillar: 0.00%
            worm: 0.00%
         leopard: 0.00%
```



## Acknowledgement
EVA-CLIP is bulit with [OpenAI CLIP](https://github.com/openai/CLIP), [Open CLIP](https://github.com/mlfoundations/open_clip) and [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark). Thanks for their awesome work!