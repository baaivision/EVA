import os
import json
import torch

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
        
def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()

def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            # if os.getenv('ENV_TYPE') == 'pytorch':
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)
    return device

def create_deepspeed_config(args):
    _, _, world_size = world_info_from_env()
    args.deepspeed_config = os.path.join(os.getcwd(), "training", "deepspeed_config.json")
    # default optimizer
    optim_settings = None
    if args.optimizer.lower() == "adamw":
        optim_settings = {
            "type": "Adam",
            "adam_w_mode": True,
            "params": {
                "bias_correction": True,
                "betas": [
                    args.beta1,
                    args.beta2
                ],
                "eps": args.eps,
            }
        }
    # LAMB
    elif args.optimizer.lower() == "lamb":
        # https://arxiv.org/pdf/1904.00962.pdf
        optim_settings = {
            "type": "LAMB",
            "params": {
            "bias_correction": True,
            "betas": [
                args.beta1,
                args.beta2
            ],
            "eps": args.eps,
            "max_coeff": 10.0, #0.3
            "min_coeff": 0.01,
            "eps_inside_sqrt": False,
            }
        }
    if args.optimizer.lower() == "1bitlamb":
        # not supported
        # 1bit-Lamb is not compatible with ZeRO; zero-stage should be 0
        # https://arxiv.org/abs/2104.06069
        optim_settings = {
            "type": "OneBitLamb",
            "params": {
            "bias_correction": True,
            "betas": [
                args.beta1,
                args.beta2
            ],
            "eps": args.eps,
            "max_coeff": 10.0, #0.3
            "min_coeff": 0.01,
            "eps_inside_sqrt": False,
            "freeze_step": args.warmup,
            # "comm_backend_name": "nccl",
            # "coeff_beta": 0.9,
            # "factor_max": 4.0,
            # "factor_min": 0.5,
            # "factor_threshold": 0.1
            }
        }

    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * world_size * args.grad_accumulation_steps,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accumulation_steps,
            "gradient_accumulation_dtype": "fp32",
            "steps_per_print": 1000,
            "zero_allow_untested_optimizer": True,
            "fp16": {
                "enabled": True if args.precision != "bf16" else False,
                # "auto_cast": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": args.precision == "bf16"
            },
            "amp": {
                "enabled": False,
                "opt_level": "O2"
            },
            "flops_profiler": {
                "enabled": True,
                "profile_step": -1,
                "module_depth": -1,
                "top_modules": 1,
                "detailed": True,
            },
            "activation_checkpointing": {
                "partition_activations": args.grad_checkpointing,
                "contiguous_memory_optimization": False,
                "profile": True
            },
            # "wallclock_breakdown": True
        }

        if optim_settings is not None:
            ds_config.update({'optimizer': optim_settings})

        if args.grad_clip_norm is not None:
            ds_config.update({'gradient_clipping': args.grad_clip_norm})

        if args.zero_stage == 1:
            ds_config.update(
                {
                    "zero_optimization": {
                        "stage": 1, 
                        "reduce_bucket_size": 5e8,
                    }
                }
            )
        elif args.zero_stage == 2:
            ds_config.update(
                {
                    "zero_optimization": {
                    "stage": 2,
                    "contiguous_gradients": ('vit-b' not in args.model.lower()), # should be False if model is small,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 5e8,
                    "allgather_bucket_size": 5e8,
                    "cpu_offload": False 
                    }
                }
            )
        elif args.zero_stage == 3:
            ds_config.update(
                {
                    "zero_optimization": {
                        "stage": 3,
                        "contiguous_gradients": True,
                        "overlap_comm": True,
                        "reduce_scatter": True,
                        "reduce_bucket_size": 5e4,
                        "allgather_bucket_size": 5e4,
                        "cpu_offload": False,
                    },
                    "stage3_max_live_parameters": 1e5,
                    "stage3_max_reuse_distance": 1e5,
                }
            )
        elif args.zero_stage > 3:
            raise NotImplementedError()

        writer.write(json.dumps(ds_config, indent=2))