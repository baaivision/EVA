from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
import sys
import warnings
from collections import OrderedDict
from einops import rearrange
from torch.utils.checkpoint import checkpoint_sequential
from .utils import get_sinusoid_encoding_table, PatchEmbed3D
from .gvt import GlobalVideoTransformer

sys.path.append("../")
from clip.model import CLIP, LayerNorm, Transformer, DropPath, QuickGELU
from timm.models.layers import trunc_normal_
import clip

from .beit import VisionTransformer, LayerNormWithForceFP32
from functools import partial


class CLIP_MAE(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 num_frames=8,
                 tubelet_size=1,
                 num_classes=1000,
                 drop_path_ratio=0.,
                 init_values=0.,
                 use_cache=True,
                 use_checkpoint=False,
                 myclip_dict=None,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

        # video
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.tubelet_size = tubelet_size
        self.input_frame = num_frames // tubelet_size
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_ratio, vision_layers)] if drop_path_ratio > 0. else None
        self.vision_layers = vision_layers
        vision_heads = vision_width // 64
        self.visual = GlobalVideoTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            droppath=dpr,
            init_values=init_values,
            use_checkpoint=use_checkpoint,
            myclip_dict=myclip_dict,
            num_classes=num_classes
        )

        # text
        if myclip_dict['USE_TEXT_EMBED']:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # learnable
            self.ln_final = LayerNorm(transformer_width)
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            # cache text features
            self.use_cache = use_cache
            self.cache_text_features = None

            self.initialize_parameters()

    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        """
               text: (NUM_CLASSES, 77)
        """
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_video(self, x):
        """
        video: (bs, downsampled_frame, width, gird ** 2)
        """

        video_features = self.visual(x)

        return video_features

    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                self.cache_text_features = self.encode_text(text)
        self.train()
        return self.cache_text_features

    def forward(self, video, text=None, use_text_embed=True):
        """
        video: (bs, frame, 3, img_size, img_size)
        text: (NUM_CLASSES, 77)
        """
        b = video.shape[0]
        video_features = self.encode_video(video)

        if use_text_embed:
            if self.use_cache:
                text_features = self.cache_text(text)
            else:
                text_features = self.encode_text(text)
            text_features = text_features.unsqueeze(0).expand(b, -1, -1)

            # calculate cosine similarity
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features)

            return logits
        else:
            return video_features


def build_model(state_dict: dict, init_mode='clip', class_mapping=None,
                img_size=224, num_frames=8, tubelet_size=1, num_classes=1000,
                drop_path_ratio=0., drop_rate=0., attn_drop_rate=0., init_values=0.,
                use_cache=True, use_checkpoint=False, myclip_dict=None, logger=None):
    if init_mode == 'clip':  # create clip visual + text branches
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        model = CLIP_MAE(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            num_frames=num_frames, tubelet_size=tubelet_size, num_classes=num_classes,
            drop_path_ratio=drop_path_ratio, init_values=init_values,
            use_cache=use_cache, use_checkpoint=use_checkpoint,  myclip_dict=myclip_dict,
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        positional_embedding = state_dict["visual.positional_embedding"]
        model.visual.positional_embedding = nn.Parameter(torch.cat([positional_embedding[0, :].reshape(1, -1),
                                                                    positional_embedding[1:, :].repeat(
                                                                        model.visual.input_frame, 1)], dim=0))
        del state_dict["visual.positional_embedding"]
        logger.info(f"copy {model.input_frame} pretrained learnable_PE params!")

        # 2D patch_embed -> 3D patch_embed
        model.visual.conv1.weight = nn.Parameter(
            state_dict["visual.conv1.weight"].unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1) / tubelet_size)
        del state_dict["visual.conv1.weight"]
        logger.info(f"load pretrained Conv2d params for Conv3d!")

        if not myclip_dict['USE_TEXT_EMBED']:
            del state_dict['visual.proj']

        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"load pretrained CLIP: {msg}")

    else:  # only create visual net
        vision_width = state_dict["patch_embed.proj.weight"].shape[0]
        assert vision_width == 1408  # eva-1b/14
        model = VisionTransformer(
            img_size=img_size, num_frames=num_frames, patch_size=14, tubelet_size=tubelet_size,
            num_classes=num_classes, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408, qkv_bias=True,
            norm_layer=partial(LayerNormWithForceFP32, eps=1e-6), init_values=0.,
            drop_path_rate=drop_path_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint, use_mean_pooling=myclip_dict['USE_MEAN_POOLING'],
            stop_grad_conv1=myclip_dict['STOP_GRAD_CONV1'])

        if init_mode == 'eva':
            pos_embed = state_dict["pos_embed"]
            model.pos_embed = nn.Parameter(torch.cat([pos_embed[:, 0, :].reshape(1, 1, -1),
                                                      pos_embed[:, 1:, :].repeat(1, model.input_frame, 1)],
                                                     dim=1))
            del state_dict["pos_embed"]
            logger.info(f"copy {model.input_frame} pretrained learnable_PE params!")

            # 2D patch_embed -> 3D patch_embed
            model.patch_embed.proj.weight = nn.Parameter(
                state_dict["patch_embed.proj.weight"].unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1) / tubelet_size)
            model.patch_embed.proj.bias = nn.Parameter(state_dict["patch_embed.proj.bias"])
            del state_dict["patch_embed.proj.weight"]
            del state_dict["patch_embed.proj.bias"]
            logger.info(f"load pretrained Conv2d params for Conv3d!")

        for k in ['head.weight', 'head.bias']:
            if k in state_dict and state_dict[k].shape != model.state_dict()[k].shape:
                if class_mapping is not None:
                    mapping = np.load(class_mapping)
                    mask = torch.from_numpy(mapping)
                    state_dict[k] = state_dict[k][mask]
                    logger.info(f"Loading masked key {k} from {class_mapping}")
                else:
                    logger.info(f"Removing key {k} from pretrained checkpoint")
                    del state_dict[k]

        # extend to more frame & larger resolution training
        pos_embed = state_dict["pos_embed"]
        num_extra_tokens = myclip_dict['NUM_EXTRA_TOKENS']
        embedding_size = vision_width
        orig_size = myclip_dict['FT_IMAGE_SIZE'] // 14
        new_size = img_size // 14

        orig_frame = (pos_embed.shape[1] - num_extra_tokens) // orig_size // orig_size
        assert orig_frame == myclip_dict['FT_FRAMES']

        if orig_frame != model.input_frame or orig_size != new_size:
            extra_tokens = pos_embed[:, :num_extra_tokens]
            pos_tokens = pos_embed[:, num_extra_tokens:]

            pos_tokens = pos_tokens.permute(0, 2, 1)

            pos_tokens = pos_tokens.view(-1, embedding_size, orig_frame, orig_size, orig_size)

            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens.float(), size=(model.input_frame, new_size, new_size), mode='trilinear',
                align_corners=False)

            pos_tokens = pos_tokens.view(pos_tokens.shape[0], embedding_size, -1)
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

            state_dict["pos_embed"] = new_pos_embed

        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loading pretrained: {msg}")

    return model.eval()


def load(model_path, model_arch: str, init_mode: str, class_mapping=None,
         device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=True,
         img_size=224, num_frames=8, tubelet_size=1, num_classes=1000,
         drop_path_ratio=0., drop_rate=0., attn_drop_rate=0., init_values=0.,
         use_cache=True, use_checkpoint=False, myclip_dict=None, logger=None):
    if init_mode == 'clip':  # clip pretrained models
        assert model_arch in clip._MODELS

        if model_path is None:
            model_path = clip._download(clip._MODELS[model_arch])
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(model_path, map_location="cpu")
        state_dict = state_dict or model.state_dict()
    else:  # eva or eva_k722 pretrained models
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'module' in checkpoint:
            state_dict = checkpoint['module']

    model = build_model(state_dict, init_mode=init_mode, class_mapping=class_mapping,
                        img_size=img_size, num_frames=num_frames, tubelet_size=tubelet_size, num_classes=num_classes,
                        drop_path_ratio=drop_path_ratio, drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate, init_values=init_values,
                        use_cache=use_cache, use_checkpoint=use_checkpoint,
                        myclip_dict=myclip_dict, logger=logger)

    if str(device) == "cpu":
        model.float()

    return model, model.state_dict()
