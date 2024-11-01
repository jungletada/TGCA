import os
import sys
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

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import utils
from engine import compute_mAP
from utils import str2bool
from datasets_cam import build_dataset
import net.simplevit

import warnings
warnings.filterwarnings("ignore")

 
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_per_gpu', default=16, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--work_space", default="results_voc", type=str)
    
    # ddp settings
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')  
    parser.add_argument('--gpu_id', default=0, type=int, help="which gpu to use")
    parser.add_argument("--local_rank", type=int, help='rank in current node')  
    parser.add_argument('--device', default='cuda',help='device id (i.e. 0 or 0,1 or cpu)')

    # Model parameters
    parser.add_argument('--model', default='ResNet38d_patch_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=448, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
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
                        help='Use Auto Augment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
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

    # Dataset parameters
    parser.add_argument('--dataset', default='', type=str, help='name of dataset')
    parser.add_argument('--voc12_root', default='datasets/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument("--coco_root", default='datasets/MSCOCO', type=str, help="Path to MSCOCO")
    parser.add_argument("--train_list", default="configs/voc12/train_aug_id.txt", type=str, 
                        help='configs/coco/train_id.txt or configs/voc12/train_aug_id.txt')
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
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)
    dist.barrier()
      

def ddp_print(logger, log_msg):
     if dist.get_rank() == 0:
         logger.info(log_msg)


def train_one_epoch(model, data_loader, optimizer, device, epoch,
                    loss_scaler, max_norm, set_training_mode=True):
    print_freq = 10
    model.train(set_training_mode)
    criterion = nn.MultiLabelSoftMarginLoss(weight=None)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)

            total_loss = criterion(outputs, targets)
            metric_logger.update(loss=total_loss.item())
                        
        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    if dist.get_rank() == 0:
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}   
   
     
@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    mAP = []
    model.eval() # switch to evaluation mode

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            cls_out = model(images)
            loss = criterion(cls_out, target)
            
            cls_out = torch.sigmoid(cls_out)
            mAP_list = compute_mAP(target, cls_out)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)
            
        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* mAP {mAP.global_avg:.3f}, loss {losses.global_avg:.3f}'
        .format(mAP=metric_logger.mAP, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

     
def main(args):
    session_name = 'Train-CAM'
    args = parser.parse_args()
    init_distributed_mode(args)   
    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)
    same_seeds(args.seed)
    # Train and Validation for image classification
    dataset_train, args.nb_classes = build_dataset(
        is_train=True, make_cam=False, args=args)
    
    dataset_val, _ = build_dataset(is_train=False, make_cam=False, args=args)
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
        batch_size=int(1.5 * args.batch_per_gpu),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False)
    
    model = create_model(
        args.model,
        pretrained=None,
        num_classes=args.nb_classes)
   
    best_ckpt_name = f'{args.model}_best.pth'
    utils.data_mkdir(args.work_space)
    
    utils.logger_info(logger_name=session_name, 
                      log_path=os.path.join(args.work_space, f'train_cam_{args.dataset}.log'))
    logger = logging.getLogger(session_name)
    ddp_print(logger, f"Use seed: {args.seed}")
    
    model.load_state_dict(checkpoint_model, strict=False)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ddp_print(logger, f'Number of parameters: {n_parameters}')
    ddp_print(logger, best_ckpt_name)

    linear_scaled_lr = args.lr * args.batch_per_gpu * dist.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    ddp_print(logger, f"|-- Total epochs: {args.epochs}")
    start_time = time.time()
    max_accuracy = 0.0

    model.to(device)
    if args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    model = nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=False, device_ids=[args.local_rank])
    
    for epoch in range(args.start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model, 
            data_loader=data_loader_train,
            optimizer=optimizer, 
            device=device, 
            epoch=epoch, 
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad)

        lr_scheduler.step(epoch)

        test_stats = evaluate(
            model=model, 
            data_loader=data_loader_val, 
            device=device)
    
        ddp_print(logger, f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%")
        if test_stats["mAP"] > max_accuracy:
            torch.save({'model': model.module.state_dict()}, 
                       os.path.join(args.work_space, f'{args.model}_best.pth'))

        max_accuracy = max(max_accuracy, test_stats["mAP"])
        ddp_print(logger, f'Max mAP: {max_accuracy * 100:.2f}%')

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()}}

        if utils.is_main_process():
            ddp_print(logger, json.dumps(log_stats))

    torch.save({'model': model.module.state_dict()}, os.path.join(args.work_space, f'{args.model}_last.pth'))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    ddp_print(logger, 'Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DeiT training and evaluation script', 
        parents=[get_args_parser()])
    
    args = parser.parse_args()
    main(args)
