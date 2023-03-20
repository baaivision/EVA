import numpy
import numpy as np
import torch.distributed as dist
import torch
import clip
import os
from apex import amp


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, amp_state_dict, logger, working_dir,
                 is_best=False):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'amp': amp_state_dict,
                  'config': config}

    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME):
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        for k in ['head.weight', 'head.bias']:
            if k in load_state_dict and load_state_dict[k].shape != model.state_dict()[k].shape:
                if config.TEST.K722_CLASS_MAPPING is not None:
                    mapping = np.load(config.DATA.K722_CLASS_MAPPING)
                    mask = torch.from_numpy(mapping)
                    load_state_dict[k] = load_state_dict[k][mask]
                    logger.info(f"Loading masked key {k} from {config.TEST.K722_CLASS_MAPPING}")
                else:
                    logger.info(f"Removing key {k} from pretrained checkpoint")
                    del load_state_dict[k]

        pos_embed = load_state_dict["pos_embed"]
        num_extra_tokens = config.MYCLIP['NUM_EXTRA_TOKENS']
        embedding_size = 1408
        orig_size = config.MYCLIP['FT_IMAGE_SIZE'] // 14
        new_size = config.DATA.INPUT_SIZE // 14

        orig_frame = (pos_embed.shape[1] - num_extra_tokens) // orig_size // orig_size
        assert orig_frame == config.MYCLIP['FT_FRAMES']

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

            load_state_dict["pos_embed"] = new_pos_embed

        msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            if 'amp' in checkpoint:
                amp.load_state_dict(checkpoint['amp'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")

            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            logger.info(f"=> loaded unsuccessfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes])

    return classes
