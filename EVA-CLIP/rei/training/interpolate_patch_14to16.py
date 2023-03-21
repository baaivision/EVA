# --------------------------------------------------------
# EVA: Exploring the Limits of Masked Visual Representation Learning at Scale (https://arxiv.org/abs/2211.07636)
# Github source: https://github.com/baaivision/EVA
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
# Based on timm, DINO, DeiT and BEiT codebases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------'

import argparse
import torch


def interpolate_pos_embed(checkpoint_model, new_size=16):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = 196
        num_extra_tokens = 1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='interpolate patch_embed kernel')
    parser.add_argument('--input', default='/path/to/eva_psz14.pt', type=str, metavar='PATH', required=True,
                        help='path to input EVA checkpoint with patch_embed kernel_size=14x14')
    parser.add_argument('--output', default='/path/to/eva_psz14to16.pt', type=str, metavar='PATH', required=True,
                        help='path to output EVA checkpoint with patch_embed kernel_size=16x16')
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location=torch.device("cpu"))
    
    # interpolate patch_embed
    patch_embed = checkpoint['patch_embed.proj.weight']
    C_o, C_in, H, W = patch_embed.shape
    patch_embed = torch.nn.functional.interpolate(
        patch_embed.float(), size=(16, 16), mode='bicubic', align_corners=False)
    checkpoint['patch_embed.proj.weight'] = patch_embed

    # interpolate pos_embed too
    interpolate_pos_embed(checkpoint, new_size=16)

    print('======== new state_dict ========')
    for k, v in list(checkpoint.items()):
        print(k, '        ', v.shape)

    torch.save(checkpoint, args.output)