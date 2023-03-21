from math import pi
import torch
from torch import nn
from einops import rearrange, repeat


# pe:16x16 -> 24x24 (224 pixel->336 pixel)
def interpolate_pos_embed(checkpoint):
    state_dict = torch.load(checkpoint, map_location='cpu')
    checkpoint_model = state_dict['module']
    
    if 'visual.pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['visual.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = (336//14)**2
        num_extra_tokens = 1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5) # 16
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5) # 24
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
        checkpoint_model['visual.pos_embed'] = new_pos_embed

        # patch_embed_proj = state_dict['visual.patch_embed.proj.weight']
        # patch_size = 14
        # state_dict['visual.patch_embed.proj.weight'] = torch.nn.functional.interpolate(
        #     patch_embed_proj.float(), size=patch_size, mode='bicubic', align_corners=False)
    
        state_dict['module'] = checkpoint_model
        torch.save(state_dict, checkpoint.replace('states', 'states_interp'))

if __name__ == '__main__':
    interpolate_pos_embed(/path/to/model_psz_14_224_ckpt)