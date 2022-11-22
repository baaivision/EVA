import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from tkinter import E
from typing import Optional, Tuple, Any, Union, List

import torch
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop

from eva_model import EVA_CLIP, convert_weights_to_fp16

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry

def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

def load_state_dict(checkpoint_path: str, map_location: str='cpu', model_key='model|module|state_dict'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    for mk in model_key.split('|'):
        if isinstance(checkpoint, dict) and mk in checkpoint:
            state_dict = checkpoint[mk]
            break
        else:
            state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(model, checkpoint_path, model_key="model|module|state_dict", strict=True):
    state_dict = load_state_dict(checkpoint_path, model_key=model_key)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    print(incompatible_keys)
    return incompatible_keys

def create_model(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        force_quick_gelu: bool = False,
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names

    if model_name in _MODEL_CONFIGS:
        logging.info(f'Loading {model_name} model config.')
        model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    model = EVA_CLIP(**model_cfg)

    load_checkpoint(model, pretrained)
                
    model.to(device=device)
    if precision == "fp16":
        assert device.type != 'cpu'
        convert_weights_to_fp16(model)

    # set image / mean metadata from pretrained_cfg if available, or use default
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD

    return model

def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    
    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)

def build_eva_model_and_transforms(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        force_quick_gelu: bool = False,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
):
    model = create_model(
        model_name, pretrained, precision, device,
        force_quick_gelu=force_quick_gelu)

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_val = image_transform(model.visual.image_size, mean=image_mean, std=image_std)

    return model, preprocess_val