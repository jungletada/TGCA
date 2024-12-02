import os
import math
import argparse
import datetime
import time
import json
import random
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import utils
from engine import evaluate
from engine import train_one_epoch_mctformerplus, \
    train_one_epoch_multioutputs
from datasets_cam import build_dataset

import net.srmct
import net.mct_adapter
import net.mctformer_plus
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_per_gpu', default=16, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument("--work_space", default="results/", type=str)
    
    # ddp settings
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')  
    parser.add_argument('--gpu_id', default=0, type=int, help="which gpu to use")
    parser.add_argument("--local_rank", type=int, help='rank in current node')  
    parser.add_argument('--device', default='cuda',help='device id (i.e. 0 or 0,1 or cpu)')

    # Model parameters
    parser.add_argument('--model', default=None, type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=448, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--cls_weight', type=float, default=3.0,
                        help='weight for class output loss')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=5e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=int, default=10, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use Auto Augment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', 
                        default='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth', 
                        help='finetune from checkpoint')
    parser.add_argument('--resume',
                        default='results_coco/mcta/msgformer_last_ckpt.pth',
                        help='resume from checkpoint')

    # Dataset parameters
    parser.add_argument('--dataset', default='', type=str, help='name of dataset')
    parser.add_argument('--voc12_root', default='data/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument("--coco_root", default='data/MSCOCO', type=str, help="Path to MSCOCO")
    parser.add_argument("--train_list", default="train_aug_id.txt", type=str, 
                        help='train_id.txt or train_aug_id.txt')
    parser.add_argument('--log_dir', default='log_dir', type=str, 
                        help='log dir to save the results')
    parser.add_argument('--checkpoint', default='', help='checkpoint for generating maps')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    return parser


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if seed == 0:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    args.dist_url = 'env://'
    
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)
    dist.barrier()

   
def load_model_weight(args, model):
    """Load model weights from a checkpoint or URL.

    Args:
        args: Command line arguments containing model configuration.
        model: The model instance to load weights into.

    Returns:
        A state dictionary of the model with loaded weights.
    """
    nc = args.nb_classes
    model_npatches = model.patch_embed.num_patches
    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
        num_extra_tokens = 1
    else: 
        checkpoint = torch.load(args.finetune, map_location='cpu')
        num_extra_tokens = model.pos_embed.shape[-2] - model_npatches
        
    try: 
        checkpoint_model = checkpoint['model']
    except KeyError:  # Specify the exception type
        checkpoint_model = checkpoint
        
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    ckpt_pos_embed = checkpoint_model['pos_embed']
    embedding_size = ckpt_pos_embed.shape[-1]

    original_size = int((ckpt_pos_embed.shape[-2] - num_extra_tokens) ** 0.5)
    if args.finetune.startswith('https'):
        extra_tokens = ckpt_pos_embed[:, :num_extra_tokens].repeat(1, nc, 1)
    else:
        extra_tokens = ckpt_pos_embed[:, :num_extra_tokens]
    pos_tokens = ckpt_pos_embed[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape( # (1, Hp, Wp, C)->(1, C, Hp, Wp)
        -1, original_size, original_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(
            input=pos_tokens,
            size=(model.Hp, model.Wp),
            mode='bicubic',
            align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

    checkpoint_model['pos_embed_cls'] = extra_tokens
    checkpoint_model['pos_embed_pat'] = pos_tokens

    if args.finetune.startswith('https'):
        cls_token_checkpoint = checkpoint_model['cls_token']
        new_cls_token = cls_token_checkpoint.repeat(1, nc, 1)
        checkpoint_model['cls_token'] = new_cls_token
    
    return checkpoint_model


def main(args):
    """
    Main function to train and evaluate the model.

    Args:
        args: Command line arguments containing model configuration and training parameters.
    """
    session_name = 'Training for Classification'
    init_distributed_mode(args)
    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)
    if args.seed is None:
        args.seed = random.randint(0, 3410)
    same_seeds(args.seed)
    # Train and Validation for image classification
    dataset_train, args.nb_classes = build_dataset(
        is_train=True, make_cam=False, args=args)
    dataset_val, _ = build_dataset(
        is_train=False, make_cam=False, args=args)
    sampler_train = DistributedSampler(dataset_train)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_per_gpu,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)

    sampler_val = DistributedSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=int(2 * args.batch_per_gpu),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False)
    
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        input_size=args.input_size)

    best_ckpt_name = f'{args.model}_best.pth'
    utils.data_mkdir(args.work_space)
    args.log_dir = os.path.join(args.work_space, args.log_dir)
    utils.data_mkdir(args.log_dir)
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    utils.logger_info(logger_name=session_name, log_path=os.path.join(
                      args.log_dir, f'train-{time}-{args.dataset}.log'))
    logger = logging.getLogger(session_name)

    if args.finetune:
        checkpoint_model = load_model_weight(args, model)
        model.load_state_dict(checkpoint_model, strict=False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    linear_scaled_lr = args.lr * args.batch_per_gpu * dist.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    # if args.resume is not None:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.load_state_dict(checkpoint['model'], strict=True)
    #     args.start_epoch = checkpoint['epoch']
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    lr_scheduler, _ = create_scheduler(args, optimizer)
    max_accuracy = 0.0
    if dist.get_rank() == 0:
        logger.info(
            "Use seed: %s\n"
            "Number of parameters: %d\n"
            "Checkpoint saved as %s\n"
            "|-- Total epochs: %d",
            args.seed, n_parameters, best_ckpt_name, args.epochs
        )

    model.to(device)
    if args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True, device_ids=[args.local_rank])
    
    if "mctformerplus" in args.model:
        train_one_epoch = train_one_epoch_mctformerplus
    else:
        train_one_epoch = train_one_epoch_multioutputs
    
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            args=args,
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad)

        lr_scheduler.step(epoch)
        # tlr=optimizer.param_groups[0]["lr"]
        # logger.info(f'{epoch}: {tlr:.8f}')
        
        test_stats = evaluate(
            model=model,
            data_loader=data_loader_val,
            device=device)

        if test_stats["mAP"] > max_accuracy:
            torch.save({'model': model.module.state_dict()},
                       os.path.join(args.work_space, f'{args.model}_best.pth'))

        max_accuracy = max(max_accuracy, test_stats["mAP"])

        if utils.is_main_process():
            log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()}}
            logger.info(
                f'mAP on the {len(dataset_val)} test images: {test_stats["mAP"] * 100:.1f}%\n' +
                f'Max mAP: {max_accuracy * 100:.2f}%\n' + json.dumps(log_stats)
            )
        torch.save({'model': model.module.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()},
                   os.path.join(args.work_space, f'{args.model}_last_ckpt.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DeiT training and evaluation script', 
        parents=[get_args_parser()])

    args = parser.parse_args()
    main(args)
