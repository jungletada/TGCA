import os
import torch
import random
import argparse
import importlib
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.backends import cudnn
import torch.distributed as dist
import torch.nn.functional as F
import tool.exutils as exutils
from tool import pyutils, torchutils
cudnn.enabled = True


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
        
        
def get_args_parser():
    parser = argparse.ArgumentParser()
    # ddp settings
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')  
    parser.add_argument('--gpu_id', default=0, type=int, help="which gpu to use")
    parser.add_argument("--local_rank", type=int, help='rank in current node')  
    parser.add_argument('--device', default='cuda',help='device id (i.e. 0 or 0,1 or cpu)')
    
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--list_path", default="voc12/train_aug_id.txt", type=str)
    parser.add_argument("--img_path", default="", type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--seg_pgt_path", default=None, type=str)

    parser.add_argument("--batch_per_gpu", default=4, type=int)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--network", default='resnet38_seg', type=str)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--init_weights", default='', type=str)
    parser.add_argument("--session_name", default="resnet38_seg", type=str)
    parser.add_argument("--crop_size", default=321, type=int)
    parser.add_argument('--print_intervals', type=int, default=50)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args_parser()
    init_distributed_mode(args)
    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)
    same_seeds(args.seed)
    
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    pyutils.Logger(os.path.join(args.save_path, args.session_name + '.log'))

    criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').cuda()
    model = getattr(importlib.import_module('network.resnet38_seg'), 'ResNet38d_Seg')(num_classes=args.num_classes)
    weights_dict = torch.load(args.init_weights)
    model.load_state_dict(weights_dict, strict=False)

    img_list = exutils.read_file(args.list_path)
    train_size = len(img_list)
    args.world_size = dist.get_world_size()
    args.batch_size = args.batch_per_gpu * args.world_size
    num_batches_per_epoch = train_size // args.batch_size
    args.max_step = args.num_epochs * num_batches_per_epoch

    data_list = []
    for i in range(500):
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    optimizer = torchutils.PolyOptimizer_cls([
        {'params': model.get_1x_lr_params(), 'lr': args.lr},
        {'params': model.get_10x_lr_params(), 'lr': 10 * args.lr}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)

    model.to(device)
    if args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True, device_ids=[args.local_rank])
    model.train()

    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")

    data_gen = exutils.chunker(data_list, args.batch_size)

    for epoch in range(args.num_epochs):
        for iteration in range(num_batches_per_epoch):
            chunk = data_gen.__next__()
            images, original_images, seg_labels, img_names = exutils.get_data_from_chunk(chunk, args)
            b, _, h, w = original_images.shape
            seg_labels = seg_labels.long().to(device)
            images = images.to(device)
            
            pred = model(x=images)
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
            loss = criterion(pred, seg_labels)

            avg_meter.add({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % args.print_intervals == 0 and dist.get_rank() == 0:
                timer.update_progress(optimizer.global_step / args.max_step)
                print('Iter: [%5d/%5d]' % (optimizer.global_step - 1, args.max_step),
                    'Loss: %.4f' % (avg_meter.pop('loss')),
                    'imps: %.1f' % ((iteration + 1) * args.batch_size / timer.get_stage_elapsed()),
                    'Fin: %s' % (timer.str_est_finish()),
                    'lr: %.5f' % (optimizer.param_groups[0]['lr']), flush=True)
                
        if dist.get_rank() == 0 and (epoch + 1) % 10 == 0:
            torch.save(model.module.state_dict(), 
                    os.path.join(args.save_path, args.session_name + f'_{epoch+1}.pth')) 
                 
        if dist.get_rank() == 0 and epoch == args.num_epochs - 1:
            torch.save(model.module.state_dict(), 
                    os.path.join(args.save_path, args.session_name + '_last.pth'))