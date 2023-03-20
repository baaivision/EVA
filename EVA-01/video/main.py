import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler, LayerDecayValueAssigner
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_train_dataloader, build_val_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending, MixupBlending, CutmixBlending
from utils.config import get_config
from models import clip_mae

import wandb
import warnings

warnings.filterwarnings('ignore')


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/kinetics400_ft.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')

    parser.add_argument('--wandb', action='store_true', help="turn on wandb for logging")

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    # load data
    train_data, train_loader = build_train_dataloader(config)
    val_data, val_loader = build_val_dataloader(config)

    # create model
    model, _ = clip_mae.load(model_path=config.MODEL.PRETRAINED,
                             model_arch=config.MODEL.ARCH,
                             init_mode=config.MODEL.PRETRAINED_MODE,
                             class_mapping=config.MODEL.K722_CLASS_MAPPING,
                             device="cpu",
                             jit=False,
                             img_size=config.DATA.INPUT_SIZE,
                             num_frames=config.DATA.NUM_FRAMES,
                             tubelet_size=config.DATA.TUBELET_SIZE,
                             num_classes=config.DATA.NUM_CLASSES,
                             drop_path_ratio=config.MODEL.DROP_PATH_RATE,
                             drop_rate=config.MODEL.DROP_RATE,
                             attn_drop_rate=config.MODEL.ATTN_DROP_RATE,
                             init_values=config.MODEL.INIT_VALUES,
                             use_cache=config.MODEL.FIX_TEXT,
                             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                             myclip_dict=config.MYCLIP,
                             logger=logger,

                             )
    model = model.cuda()

    # partially (fully) fine-tune the model (default: 0)
    if config.MODEL.FIX_FINE_TUNE > 0:
        for name, param in model.named_parameters():
            if 'block' in name:
                num_block = name.split('.')[1]
                if int(num_block) < config.MODEL.FIX_FINE_TUNE:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False

        for _, param in model.head.named_parameters():
            param.requires_grad = True
        for _, param in model.fc_norm.named_parameters():
            param.requires_grad = True

    for n, p in model.named_parameters():
        logger.info(f'{n} requires_grad = {p.requires_grad}')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters / 1e6:.4f}MB')

    # create mixup cutmix label-smoothing
    mixup_fn = None
    if config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0:
        criterion = SoftTargetCrossEntropy()
        if config.AUG.MIXUP > 0 and config.AUG.CUTMIX > 0:
            mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                           smoothing=config.AUG.LABEL_SMOOTH,
                                           mixup_alpha=config.AUG.MIXUP,
                                           cutmix_alpha=config.AUG.CUTMIX,
                                           switch_prob=config.AUG.MIXUP_SWITCH_PROB)
        elif config.AUG.MIXUP > 0 and config.AUG.CUTMIX == 0:
            mixup_fn = MixupBlending(num_classes=config.DATA.NUM_CLASSES,
                                     smoothing=config.AUG.LABEL_SMOOTH,
                                     alpha=config.AUG.MIXUP,
                                     )
        else:
            mixup_fn = CutmixBlending(num_classes=config.DATA.NUM_CLASSES,
                                      smoothing=config.AUG.LABEL_SMOOTH,
                                      alpha=config.AUG.CUTMIX,
                                      )
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()

    # create optimizer & scheduler
    if config.TRAIN.LAYER_WISE_DECAY < 1.0:
        num_layers = model.vision_layers
        lr_scale_value = list(config.TRAIN.LAYER_WISE_DECAY ** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner(lr_scale_value)
    else:
        assigner = None
    if assigner is not None:
        logger.info(f'Assigned values: {assigner.values}')
    optimizer = build_optimizer(config, model, get_num_layer=assigner.get_layer_id if assigner is not None else None,
                                get_layer_scale=assigner.get_scale if assigner is not None else None)

    n_iter_per_epoch = len(train_data) // (
            config.TRAIN.BATCH_SIZE * config.TRAIN.ACCUMULATION_STEPS * int(os.environ['WORLD_SIZE']))
    lr_scheduler = build_scheduler(config, optimizer, n_iter_per_epoch=n_iter_per_epoch)

    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,
                                                      find_unused_parameters=True if config.MODEL.FIX_FINE_TUNE > 0 else False)

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

    text_labels = generate_text(train_data)

    if config.TEST.ONLY_TEST:
        acc1, acc5 = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on ({len(val_data)}) videos: {acc1:.1f}% (acc@1) / {acc5:.1f}% (acc@5)")
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn,
                        n_iter_per_epoch)

        acc1, acc5 = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on ({len(val_data)}) videos: {acc1:.1f}% (acc@1) / {acc5:.1f}% (acc@5)")

        if config.WANDB and dist.get_rank() == 0:
            wandb.log(
                {
                    'ep': epoch,
                    'acc1': acc1,
                    'acc5': acc5,
                },
                step=(epoch + 1) * n_iter_per_epoch + 1
            )

        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, amp.state_dict(),
                         logger, config.OUTPUT, is_best)

    # multi-view test
    logger.info(f"Multi views test starting!")
    config.defrost()
    config.TEST.NUM_CROP = 3
    config.TEST.NUM_CLIP = 4
    config.freeze()
    val_data, val_loader = build_val_dataloader(config)

    acc1, acc5 = validate(val_loader, text_labels, model, config)
    logger.info(f"Accuracy of the network on ({len(val_data)}) videos: {acc1:.1f}% (acc@1) / {acc5:.1f}% (acc@5)")

    # last epoch log with multi-view acc@1
    if config.WANDB and dist.get_rank() == 0:
        wandb.log(
            {
                'ep': epoch,
                'acc1': acc1,
                'acc5': acc5,
            },
            step=(config.TRAIN.EPOCHS + 1) * n_iter_per_epoch
        )


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn,
                    num_steps):
    model.train()
    optimizer.zero_grad()

    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    texts = text_labels.cuda(non_blocking=True)

    for idx, batch_data in enumerate(train_loader):

        step = idx // config.TRAIN.ACCUMULATION_STEPS

        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        images = images.view((-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)

        output = model(images, text=texts, use_text_embed=config.MYCLIP.USE_TEXT_EMBED)

        total_loss = criterion(output, label_id)
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + step)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + step)

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        lr_first = optimizer.param_groups[0]['lr']
        lr_last = optimizer.param_groups[-1]['lr']
        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx) if not is_lr_schedule else batch_time.avg * (num_steps - step)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx if not is_lr_schedule else step}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr_first {lr_first:.9f} lr_last {lr_last:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            if config.WANDB and dist.get_rank() == 0:
                wandb.log(
                    {
                        'ep': epoch,
                        'lr_first': lr_first,
                        'lr_last': lr_last,
                        'tot_loss': tot_loss_meter.val,
                        'tot_loss_avg': tot_loss_meter.avg,
                    },
                    step=(epoch * num_steps + idx) if not is_lr_schedule else (epoch * num_steps + step)
                )

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    if config.WANDB and dist.get_rank() == 0:
        wandb.log(
            {
                'ep': epoch,
                'lr_first': optimizer.param_groups[0]['lr'],
                'lr_last': optimizer.param_groups[-1]['lr'],
            },
            step=(epoch + 1) * num_steps
        )


@torch.no_grad()
def validate(val_loader, text_labels, model, config):
    model.eval()

    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP} * {config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)

            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()

                output = model(image_input, text=text_inputs, use_text_embed=config.MYCLIP.USE_TEXT_EMBED)

                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity  # summarize all views of similarity

            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                    f'Acc@5: {acc5_meter.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")

    # log with wandb
    if dist.get_rank() == 0:
        if config.WANDB:
            wandb.init(config=config, project="eva-video-finetune")
            wandb.run.name = '_'.join([
                config.MODEL.ARCH,
                config.DATA.DATASET.replace('kinetics', 'K'),
                '{}'.format(config.DATA.INPUT_SIZE),
                'f{}-{}'.format(config.DATA.NUM_FRAMES, config.DATA.TUBELET_SIZE),
                'ep{}-{}'.format(config.TRAIN.WARMUP_EPOCHS, config.TRAIN.EPOCHS),
                'lr{}'.format(config.TRAIN.LR),
                'ld{}'.format(config.TRAIN.LAYER_WISE_DECAY),
                'wd{}'.format(config.TRAIN.WEIGHT_DECAY),
                'dpr{}'.format(config.MODEL.DROP_PATH_RATE),
                'aug[ls{}-cut{}-up{}]'.format(config.AUG.LABEL_SMOOTH, config.AUG.CUTMIX, config.AUG.MIXUP),
                'bs{}*{}*{}'.format(world_size, config.TRAIN.BATCH_SIZE, config.TRAIN.ACCUMULATION_STEPS),
                's{}'.format(config.SEED)
            ])
        else:
            warnings.warn("wandb is turned off")

    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)
