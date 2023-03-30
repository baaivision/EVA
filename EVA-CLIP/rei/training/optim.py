from torch import optim
import json
import logging
import re

from .distributed import is_master
from training.adan import Adan
from training.lion import Lion
from training.lamb import Lamb
from training.anyprecision_optimizer import AnyPrecisionAdamW

try:
    from apex.optimizers import FusedAdam, FusedLAMB
except:
    print("Please install lastest apex to use FusedAdam and FusedLAMB")
    FusedAdam, FusedLAMB = None, None

def get_num_layer_for_transformer(param_name, num_max_layer):
    layer_0 = {
        "patch_embed", 
        "pos_embed", 
        "cls_token", 
        "mask_token", 
        "conv1",
        "positional_embedding",
        "token_embedding",
        "transformer.embeddings.word_embeddings",
        "transformer.embeddings.position_embeddings",
        "transformer.embeddings.token_type_embeddings",
    }

    if any(l in param_name for l in layer_0):
        return 0

    block_regex = re.compile(r"blocks\.([0-9]+)\.")
    match_block = block_regex.search(param_name)

    #huggingface->text.transformer.encoder.layer
    layer_regex = re.compile(r"layer\.([0-9]+)\.") 
    match_layer = layer_regex.search(param_name)
    if match_block is not None:
        return int(match_block.group(1)) + 1
    elif match_layer is not None:
        return int(match_layer.group(1)) + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_transformer(var_name, len(self.values))

def get_parameters(args, model, assigner, tower):
    filter_parameters = []
    skip = set()
    if tower == 'visual':
        lr = args.visual_lr if args.visual_lr is not None else args.lr
        weight_decay = args.visual_wd if args.visual_wd is not None else args.wd
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'visual.' in name]
        if hasattr(model, 'visual'):
            if hasattr(model.visual, 'no_weight_decay'):
                skip = set.union(skip, model.visual.no_weight_decay())
        skip = ['visual.' + n for n in skip]
    elif tower == 'text':
        lr = args.text_lr if args.text_lr is not None else args.lr
        weight_decay = args.text_wd if args.text_wd is not None else args.wd
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'text.' in name]
        if hasattr(model, 'text'):
            if hasattr(model.text, 'no_weight_decay'):
                skip = set.union(skip, model.text.no_weight_decay())
        skip = ['text.' + n for n in skip]
    else:
        lr = args.lr
        weight_decay = args.wd
        exclude = lambda n: 'visual.' not in n and 'text.' not in n
        filter_parameters = [[n, p] for n, p in model.named_parameters() if exclude(n)]
        if hasattr(model, 'no_weight_decay'):
            skip = set.union(skip, model.no_weight_decay())

    get_num_layer  = assigner.get_layer_id if assigner is not None else None
    get_layer_scale = assigner.get_scale if assigner is not None else None


    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in filter_parameters:
        if not param.requires_grad:
            continue

        # if param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or 'logit_scale' in name or name in skip:
        if param.ndim <= 1 or name.endswith(".bias") or name in skip:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = tower + "_" + "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr
            }
            parameter_group_vars[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    if is_master(args, local=args.log_local):
        logging.info(f"Tower = {tower}")
        logging.info(f"Skip weight decay name marked in tower-{tower}: {skip}")
        logging.info(f"Num of parameters group in tower-{tower}: {len(parameter_group_vars.values())}")
        logging.info(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
    return list(parameter_group_vars.values())


def get_assigner(args, model):
    visual_ld = args.visual_ld if args.visual_ld else args.ld
    text_ld = args.text_ld if args.text_ld else args.ld
    
    if visual_ld < 1.0:
        visual_num_layers = model.visual.get_num_layers()
        assigner_visual = LayerDecayValueAssigner(list(visual_ld ** (visual_num_layers + 1 - i) for i in range(visual_num_layers + 2)))
    else:
        assigner_visual = None

    if text_ld < 1.0:
        text_num_layers = model.text.get_num_layers()
        assigner_text = LayerDecayValueAssigner(list(text_ld ** (text_num_layers + 1 - i) for i in range(text_num_layers + 2)))
    else:
        assigner_text = None

    if assigner_visual is not None:
        logging.info("Assigned visual values = %s" % str(assigner_visual.values))
    if assigner_text is not None:
        logging.info("Assigned text values = %s" % str(assigner_text.values))
    return assigner_visual, assigner_text

def get_all_parameters(args, model):
    assigner_visual, assigner_text = get_assigner(args, model)
        
    parameters = []
    visual_parameters = get_parameters(args, model, assigner_visual, 'visual')
    text_parameters = get_parameters(args, model, assigner_text, 'text')
    other_parameters = get_parameters(args, model, None, 'other')

    parameters.extend(visual_parameters)
    parameters.extend(text_parameters)
    parameters.extend(other_parameters)

    if len(parameters) == 0:
        parameters = model.parameters()
    return parameters

def create_optimizer(args, model, return_params=False):
    optimizer_args = dict(
            betas=(args.beta1, args.beta2),
        )
    if args.optimizer != 'lion':
        optimizer_args['eps'] = args.eps
        
    if args.optimizer == 'lamb':
        base_optimizer = Lamb
    elif args.optimizer == 'adan':
        base_optimizer = Adan
    elif args.optimizer == 'fused_lamb':
        base_optimizer = FusedLAMB
    elif args.optimizer == 'lion':
        base_optimizer = Lion
    elif args.optimizer == 'ap_adamw' and args.precision == 'bf16':
        base_optimizer = AnyPrecisionAdamW
    else:
        base_optimizer = optim.AdamW

    parameters = get_all_parameters(args, model)

    optimizer = base_optimizer(parameters, **optimizer_args)

    if is_master(args, local=args.log_local):
        logging.info(f'Optimizer: {args.optimizer}')
        logging.info(f'Optimizer config: {optimizer_args}')

    if return_params:
        return optimizer, parameters
    return optimizer