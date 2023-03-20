# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on EVA: Exploring the Limits of Masked Visual Representation Learning at Scale (https://arxiv.org/abs/2211.07636)
# https://github.com/baaivision/EVA/tree/master/EVA-01
# --------------------------------------------------------'


import json
from torch.utils import data
from torchvision.datasets import ImageFolder
import torch
import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Manager
import collections
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import torchvision
import cv2

torch.manual_seed(0)
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

transform = transforms.Compose([
    transforms.Resize(448, interpolation=3),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
])

class ObjectNetDataset(ImageFolder):
    def __init__(self, imagenet_path):
        self._imagenet_path = imagenet_path
        self._all_images = []

        o_dataset = ImageFolder(self._imagenet_path)
        # get mappings folder
        mappings_folder = os.path.abspath(
            os.path.join(self._imagenet_path, "../mappings")
        )

        # get ObjectNet label to ImageNet label mapping
        with open(
            os.path.join(mappings_folder, "objectnet_to_imagenet_1k.json")
        ) as file_handle:
            o_label_to_all_i_labels = json.load(file_handle)

        # now remove double i labels to avoid confusion
        o_label_to_i_labels = {
            o_label: all_i_label.split("; ")
            for o_label, all_i_label in o_label_to_all_i_labels.items()
        }

        # some in-between mappings ...
        o_folder_to_o_idx = o_dataset.class_to_idx
        with open(
            os.path.join(mappings_folder, "folder_to_objectnet_label.json")
        ) as file_handle:
            o_folder_o_label = json.load(file_handle)

        # now get mapping from o_label to o_idx
        o_label_to_o_idx = {
            o_label: o_folder_to_o_idx[o_folder]
            for o_folder, o_label in o_folder_o_label.items()
        }

        # some in-between mappings ...
        with open(
            os.path.join(mappings_folder, "pytorch_to_imagenet_2012_id.json")
        ) as file_handle:
            i_idx_to_i_line = json.load(file_handle)
        with open(
            os.path.join(mappings_folder, "imagenet_to_label_2012_v2")
        ) as file_handle:
            i_line_to_i_label = file_handle.readlines()

        i_line_to_i_label = {
            i_line: i_label[:-1]
            for i_line, i_label in enumerate(i_line_to_i_label)
        }

        # now get mapping from i_label to i_idx
        i_label_to_i_idx = {
            i_line_to_i_label[i_line]: int(i_idx)
            for i_idx, i_line in i_idx_to_i_line.items()
        }

        # now get the final mapping of interest!!!
        o_idx_to_i_idxs = {
            o_label_to_o_idx[o_label]: [
                i_label_to_i_idx[i_label] for i_label in i_labels
            ]
            for o_label, i_labels in o_label_to_i_labels.items()
        }

        self._tag_list = []
        # now get a list of files of interest
        for filepath, o_idx in o_dataset.samples:
            if o_idx not in o_idx_to_i_idxs:
                continue
            rel_file = os.path.relpath(filepath, self._imagenet_path)
            if o_idx_to_i_idxs[o_idx][0] not in self._tag_list:
                self._tag_list.append(o_idx_to_i_idxs[o_idx][0])
            self._all_images.append((rel_file, o_idx_to_i_idxs[o_idx][0]))

    def __getitem__(self, item):
        image_path, classification = self._all_images[item]
        image_path = os.path.join(self._imagenet_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = transform(image)

        return image, classification

    def __len__(self):
        return len(self._all_images)