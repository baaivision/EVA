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


from tracemalloc import start
from typing import Iterable

import torch
import torch.nn as nn

import utils

import time

import torch.distributed as dist
import utils


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm


def compute_loss(output, label):
    loss_func = nn.CosineSimilarity(dim=-1)
    loss = loss_func(output.float(), label.float())
    return -loss.mean()


def train_one_epoch(model: torch.nn.Module, teacher: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    num_training_steps_per_epoch: int, max_norm: float = 0, update_freq: int = 1, 
                    log_writer=None, lr_scheduler=None, start_steps=None, 
                    lr_schedule_values=None, wd_schedule_values=None, beta2_values=None, args=None, 
                    mixup_fn=None, beit_like=True,
                    global_rank=1, fp16=True):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # if data_iter_step > 10: break

        start_time = time.time()

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * \
                        param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
                if beta2_values is not None:
                    param_group["betas"][1] = beta2_values[it] if it < len(beta2_values) else beta2_values[-1]
        
        samples, images, bool_masked_pos = batch    # vit, clip, mask

        if mixup_fn:
            images, _ = mixup_fn(images)
            samples = images

        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        if loss_scaler is None:
            if fp16:
                samples = samples.half()
            else:
                samples = samples.bfloat16()
            
        if beit_like:
            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_features = teacher.infer_image({"image": [images]})
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                labels = clip_features[bool_masked_pos]

            if loss_scaler is None:
                outputs = model(samples, bool_masked_pos=bool_masked_pos)
            else:
                with torch.cuda.amp.autocast():
                    outputs = model(samples, bool_masked_pos=bool_masked_pos)  

            loss = compute_loss(outputs, labels)
        else:   # mae
            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_features = teacher.infer_image({"image": [images]})

            if loss_scaler is None:
                loss = model(samples, clip_features)
            else:
                with torch.cuda.amp.autocast():
                    loss = model(samples, clip_features)

        loss_value = loss.item()

        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss)
        loss_list = torch.tensor(loss_list)

        all_loss_mean_value = loss_list.mean().item()
        metric_logger.update(all_loss_mean=all_loss_mean_value)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            print(" ========== loss_isnan = {},  loss_isinf = {} ========== ".format(loss_list_isnan, loss_list_isinf))
            if args.output_dir and args.auto_resume_iter:
                utils.auto_load_model_iter(args=args, model=model)
                continue
            else:
                exit()

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
            loss_scale_value, grad_norm = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        end_time = time.time()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        momentum = 1.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
            momentum = min(momentum, group["betas"][1])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(momentum=momentum)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(all_rank_loss_mean=all_loss_mean_value, head="loss")
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(momentum=momentum, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(time=end_time - start_time, head="time")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate_pt(data_loader, model, teacher, device, beit_like=True):
    criterion = nn.CosineSimilarity(dim=-1)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        if beit_like:
            images = batch[0]
            images = images.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                clip_features = teacher.infer_image({"image": [images]})

                bool_masked_pos = clip_features[:, :, 0].clone()
                bool_masked_pos[:, 0::2] = 1
                bool_masked_pos[:, 1::2] = 0
                bool_masked_pos = bool_masked_pos.to(torch.bool)

                labels = clip_features[bool_masked_pos]

                outputs = model(images, bool_masked_pos=bool_masked_pos)
                loss = compute_loss(outputs, labels)
        else:
            with torch.no_grad():
                images = batch[0]
                images = images.to(device, non_blocking=True)
                clip_features = teacher.infer_image({"image": [images]})

            with torch.cuda.amp.autocast():
                loss = model(images, clip_features)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
