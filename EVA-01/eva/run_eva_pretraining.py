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
import datetime
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from mixup import Mixup
from timm.models import create_model
from optim_factory import create_optimizer, get_parameter_groups

from datasets import build_eva_pretraining_dataset, build_val_dataset_for_pt
from engine_for_pretraining import train_one_epoch, evaluate_pt
from utils import NativeScalerWithGradNormCount as NativeScaler
from clip_wrapper import CLIPWrapper
import utils
import modeling_pretrain, modeling_mae_pretrain


def get_args():
    parser = argparse.ArgumentParser('EVA pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)

    # CLIP teacher setting
    parser.add_argument('--teacher_type', type=str, default='clip')
    parser.add_argument('--teacher_model_path', type=str)

    parser.add_argument('--clip_model', type=str, default='ViT-L/14')
    parser.add_argument('--cache_dir', type=str, default='/sharefs/yxf/weights/clip/large')

    # Model parameters
    parser.add_argument('--model', default='eva_g_patch14_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mae', action='store_true', help='use mae-like (enc & dec) to pre-train')

    parser.add_argument('--grad_ckpt', action='store_true')
    parser.add_argument('--stop_grad_conv1', action='store_true')

    parser.add_argument('--rel_pos_bias', action='store_true')

    parser.add_argument('--decoupled_rel_pos_bias', action='store_true')
    parser.add_argument('--disable_decoupled_rel_pos_bias', action='store_false', dest='decoupled_rel_pos_bias')
    parser.set_defaults(decoupled_rel_pos_bias=False)

    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.add_argument('--disable_abs_pos_emb', action='store_false', dest='abs_pos_emb')
    parser.set_defaults(abs_pos_emb=True)

    parser.add_argument('--layer_scale_init_value', default=0, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--num_mask_patches', default=105, type=int,
                        help='number of the visual tokens/patches need be masked')  ### 224*224/psz/psz * 0.4 ###
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=224, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-6, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.98], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr_sched_type', type=str, default="cos")

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--model_ema', default=False, action='store_true')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='bicubic',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--crop_scale', default=[0.2, 1.0], type=float, nargs='+',
                        help='Crop scale for RRC')
    parser.add_argument('--crop_ratio', default=[3. / 4., 4. / 3.], type=float, nargs='+',
                        help='Crop ratio for RRC')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Fine-tuning params
    parser.add_argument('--finetune', default='', 
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_set', default='image_folder',
                        choices=[
                            'CIFAR', 'IMNET', 'image_folder'
                        ],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--val_data_set', default='IMNET',
                        choices=[
                            'CIFAR', 'IMNET', 'image_folder'
                        ],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--crop_pct', type=float, default=None)

    parser.add_argument('--data_path', default='/path/to/merge4pt', type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default='/path/to/ImageNet-1K', type=str,
                        help='dataset path')

    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')
    parser.add_argument('--bf16', default=False, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--seed', default=88, type=int)
    parser.add_argument('--rand', action='store_true')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--auto_resume_iter', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed',
                        action='store_true', default=False)
    parser.add_argument('--zero_stage', default=1, type=int,
                        help='ZeRO optimizer stage (default: 0)')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please install DeepSpeed")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_model(args):
    print(f"Creating model: {args.model}")
    if args.mae:
        model = modeling_mae_pretrain.__dict__[args.model](
            predict_feature_dim=args.teacher_out_feat_dim,
            grad_ckpt=args.grad_ckpt
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            img_size=args.input_size,
            predict_feature_dim=args.teacher_out_feat_dim,
            drop_path_rate=args.drop_path,
            grad_ckpt=args.grad_ckpt,
            stop_grad_conv1=args.stop_grad_conv1,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_shared_decoupled_rel_pos_bias=args.decoupled_rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
        )
    return model


def main(args, ds_init):

    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    if args.rand:
        args.seed = random.randint(0, 1e6)
        print('Use rand seed', args.seed)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.teacher_type == 'clip':
        teacher = CLIPWrapper(
            clip_model=args.clip_model, 
            download_root=args.cache_dir
        )
    else:
        raise NotImplementedError()
    
    teacher = teacher.to(device)

    print("teacher = %s" % str(teacher))

    args.teacher_out_feat_dim = teacher.net.visual.output_dim

    print('teacher_out_feat_dim', args.teacher_out_feat_dim)

    model = get_model(args)
    patch_size = model.get_final_patch_size()
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_eva_pretraining_dataset(args)

    # val dataset for val loss monitor
    dataset_val, _ = build_val_dataset_for_pt(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks // args.update_freq

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank,
            shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        # val set
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, 
            cutmix_alpha=args.cutmix, 
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, 
            switch_prob=args.mixup_switch_prob, 
            mode=args.mixup_mode,
        )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        else:
            print("%s not found!" % args.finetune)

        utils.load_state_dict(model, checkpoint_model)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (B):', n_parameters / 1e9)

    total_batch_size = args.batch_size * utils.get_world_size() * args.update_freq
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, model.no_weight_decay()
        )
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp)
        loss_scaler = NativeScaler()
    
    print("optimizer = %s" % str(optimizer))

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        sched_type=args.lr_sched_type,
    )
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, 
            teacher, 
            data_loader_train,
            optimizer, 
            device, 
            epoch, 
            loss_scaler,
            max_norm=args.clip_grad, 
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, 
            wd_schedule_values=wd_schedule_values,
            update_freq=args.update_freq, 
            num_training_steps_per_epoch=num_training_steps_per_epoch, 
            mixup_fn=mixup_fn, 
            beit_like=not args.mae,
            global_rank=global_rank,
            fp16=not args.bf16,
            args=args
        )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        
        test_stats = evaluate_pt(data_loader_val, model, teacher, device, beit_like=not args.mae)
        print(f"Val loss of the network on the {len(dataset_val)} test images: {test_stats['loss']:.1f}%")

        if log_writer is not None:
            log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)