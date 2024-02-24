import os
import torch
import random
import argparse
import importlib
import sys,os
import numpy as np
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tool import pyutils, imutils, torchutils
from network.transform import Normalize

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from data.voc12.dataloader_psa import VOC12AffDatasetCRF
cudnn.enabled = True


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="res38_aff", type=str)
    # model weights and path to CAM seeds 
    parser.add_argument("--work_space", default="results/MCTG", type=str)
    parser.add_argument("--cam_out_dir", default="cam_mask", type=str)
    parser.add_argument("--train_list", default="configs/voc12/train_aug.txt", type=str)
    parser.add_argument("--model_weights", default='checkpoints/res38_cls.pth', type=str)
    parser.add_argument("--voc12_root", default='datasets/VOCdevkit/VOC2012/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    # ddp settings
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')  
    parser.add_argument('--gpu_id', default=0, type=int, help="which gpu to use")
    parser.add_argument("--local_rank", type=int, help='rank in current node')  
    parser.add_argument('--device', default='cuda',help='device id (i.e. 0 or 0,1 or cpu)')
    # training settings
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--network", default="network.resnet38_aff", type=str)
    parser.add_argument("--batch_per_gpu", default=4, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    # dataset settings
    parser.add_argument("--dataset", default="VOC", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--low_alpha", default=1, type=int)
    parser.add_argument("--high_alpha", default=12, type=int)
    parser.add_argument("--radius", default=5, type=int)

    args = parser.parse_args()
    return args


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
        

def build_psa_dataset(args):
    train_dataset = VOC12AffDatasetCRF(
        img_name_list_path=args.train_list, 
        cam_npy_dir=args.cam_out_dir,
        voc12_root=args.voc12_root, 
        cropsize=args.crop_size, 
        low_alpha=args.low_alpha, 
        high_alpha=args.high_alpha,
        radius=args.radius,
        joint_transform_list=[
            None,
            None,
            imutils.RandomCrop(args.crop_size),
            imutils.RandomHorizontalFlip()],
        img_transform_list=[
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            np.asarray,
            Normalize(),
            imutils.HWC_to_CHW],
        label_transform_list=[
            None,
            None,
            None,
            imutils.AvgPool2d(8)])
        
    return train_dataset


def build_dataloader(train_dataset, args):
    sampler_train = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=args.batch_per_gpu, 
        num_workers=args.num_workers,
        pin_memory=True, 
        drop_last=True)
    return sampler_train, train_data_loader
    
    
def print_info(args):
    import pprint
    args_dict = vars(args)
    if dist.get_rank() == 0:
        pprint.pprint(args_dict)
        
        
if __name__ == '__main__':
    args = get_args_parser() 
    
    init_distributed_mode(args)
    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)
    same_seeds(args.seed)
    
    args.num_classes = 20
    work_dir = os.path.abspath('.')
    args.cam_out_dir = os.path.join(args.work_space, args.cam_out_dir) 

    pyutils.Logger(os.path.join(args.work_space, args.session_name + '.log'))
    
    train_dataset = build_psa_dataset(args)
    sampler_train, train_data_loader = build_dataloader(train_dataset, args)
    
    args.world_size = dist.get_world_size()
    args.batch_size = args.batch_per_gpu * args.world_size
    max_step = len(train_dataset) * args.epoch // args.batch_size 
    args.max_step = max_step
    
    print_info(args)
    
    model = getattr(importlib.import_module(args.network), 'Net')()
    # load model model_weights of resnet38d
    if args.model_weights[-7:] == '.params':
        import network.resnet38d
        assert args.network == "network.resnet38_aff"
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.model_weights)
    else: weights_dict = torch.load(args.model_weights)
    
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    model.load_state_dict(weights_dict, strict=False)
    model.to(device)
    
    if args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True, 
        device_ids=[args.local_rank])
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'bg_loss', 'fg_loss', 'neg_loss',
                                     'bg_cnt', 'fg_cnt', 'neg_cnt')
    timer = pyutils.Timer("Session started: ")
    eps = 1e-6
    
    for epoch in range(args.epoch):
        sampler_train.set_epoch(epoch)
        for iteration, pack in enumerate(train_data_loader):
            aff = model.forward(pack[0]) # B x 3 x H x W
            bg_label = pack[1][0].cuda(non_blocking=True)
            fg_label = pack[1][1].cuda(non_blocking=True)
            neg_label = pack[1][2].cuda(non_blocking=True)

            bg_count = torch.sum(bg_label) + eps
            fg_count = torch.sum(fg_label) + eps
            neg_count = torch.sum(neg_label) + eps

            bg_loss = torch.sum(- bg_label * torch.log(aff + eps)) / bg_count
            fg_loss = torch.sum(- fg_label * torch.log(aff + eps)) / fg_count
            neg_loss = torch.sum(- neg_label * torch.log(1. + eps - aff)) / neg_count

            loss = bg_loss / 4 + fg_loss / 4 + neg_loss / 2
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({
                'loss': loss.item(),
                'bg_loss': bg_loss.item(), 'fg_loss': fg_loss.item(), 'neg_loss': neg_loss.item(),
                'bg_cnt': bg_count.item(), 'fg_cnt': fg_count.item(), 'neg_cnt': neg_count.item()
            })
            
            torch.cuda.synchronize()   
             
            if (optimizer.global_step - 1) % 50 == 0 and dist.get_rank() == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('Iter: [%5d/%5d]' % (optimizer.global_step-1, max_step),
                      'loss:%.4f, bg_loss:%.4f, fg_loss:%.4f, neg_loss:%.4f;' % avg_meter.get('loss', 'bg_loss', 'fg_loss', 'neg_loss'),
                      'cnt: %.0f %.0f %.0f;' % avg_meter.get('bg_cnt', 'fg_cnt', 'neg_cnt'),
                      'imps: %.1f;' % ((iteration + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin: %s;' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
                avg_meter.pop()
                
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), os.path.join(args.work_space, args.session_name+'_last.pth'))
