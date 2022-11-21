import copy
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.distributed as dist
import json


def is_main_process():
    return dist.get_rank() == 0


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("visual.class_embedding", "visual.positional_embedding", "cls_token", "mask_token", "pos_embed"):
        # class_embedding, positional_embedding, patch_embed
        return 0
    elif var_name.startswith("visual.conv1") or var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("visual.transformer.resblocks"):
        layer_id = int(var_name.split('.')[3])
        return layer_id + 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(
                param)  # param.shape==1 (include visual.class_embedding) or bias or skip_list or skip_keywords
        else:
            has_decay.append(param)

    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}]


def fix_text(model):
    for name, param in model.named_parameters():
        if name.startswith("transformer") or name.startswith("token_embedding") \
                or name.startswith("positional_embedding") or name.startswith("text_projection") \
                or name.startswith("logit_scale") or name.startswith("ln_final"):
            param.requires_grad = False
        else:
            continue


def get_parameter_groups(model, skip_list=(), skip_keywords=(), weight_decay=0.001, get_num_layer=None,
                         get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or check_keywords_in_name(name,
                                                                                                            skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())


def build_optimizer(config, model, get_num_layer=None, get_layer_scale=None):
    model = model.module if hasattr(model, 'module') else model

    # fix text
    if config.MODEL.FIX_TEXT:
        fix_text(model)

    # set decay and lr
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = get_parameter_groups(model, skip, skip_keywords,
                                      weight_decay=config.TRAIN.WEIGHT_DECAY,
                                      get_num_layer=get_num_layer, get_layer_scale=get_layer_scale)

    optimizer = optim.AdamW(parameters, betas=config.TRAIN.BETAS, eps=config.TRAIN.EPS, lr=config.TRAIN.LR)

    return optimizer


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER_WARMUP_PREFIX else num_steps,
        lr_min=config.TRAIN.LR / 100 if config.TRAIN.LR_MIN == 0. else config.TRAIN.LR_MIN,
        warmup_lr_init=config.TRAIN.WARMUP_START_LR,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
        warmup_prefix=config.TRAIN.LR_SCHEDULER_WARMUP_PREFIX,
    )

    return lr_scheduler
