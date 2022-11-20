import clip
import torch.nn as nn


class CLIPWrapper(nn.Module):
    def __init__(self, clip_model="ViT-L/16", download_root=None):
        super().__init__()
        self.net, _ = clip.load(
            clip_model, device='cpu', jit=False, 
            download_root=download_root, 
        )

    def infer_image(self, features):
        x = features["image"][0]
        x = self.net.encode_image(x)
        return x

