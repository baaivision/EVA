#!/bin/bash

# Instructions for Manual Download:
# Automatic Downloads:
set -e  # exit when any command fails
mkdir -p pretrained/
cd pretrained/
wget https://huggingface.co/BAAI/EVA/resolve/main/eva_psz14.pt        # eva_psz14 weights
wget https://huggingface.co/BAAI/EVA/resolve/main/eva_video_k722.pth  # eva_video_k722 weights
wget https://huggingface.co/BAAI/EVA/resolve/main/eva_video_k400.pth  # eva_video_k400 weights
wget https://huggingface.co/BAAI/EVA/resolve/main/eva_video_k600.pth  # eva_video_k600 weights
wget https://huggingface.co/BAAI/EVA/resolve/main/eva_video_k700.pth  # eva_video_k700 weights
