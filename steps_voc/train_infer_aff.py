import os, sys
import random
import pprint
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import PIL.Image as Image

import torch
import torch.nn as nn
import torchvision
from torch.backends import cudnn
from torch import multiprocessing, cuda
from torchvision import transforms
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


sys.path.append(osp.dirname(__file__) + os.sep + '../')
from data.voc12.dataloader_psa import VOC12AffDatasetCRF
from data.voc12.dataloader_psa import VOC12ImageDataset
from models.resnet38_aff import ResNet38d_Aff
from models.tool import pyutils, imutils, torchutils
from misc.torchutils import split_dataset 
from utils import str2bool, data_mkdir

cudnn.enabled = True


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, type=str2bool)
    parser.add_argument("--inference", default=False, type=str2bool)
    # model weights and path to CAM seeds 
    parser.add_argument("--work_space", default="results/MCTG", type=str)
    parser.add_argument("--cam_out_dir", default="cam_mask", type=str, help="cam mask path")
    parser.add_argument("--seg_out_dir", default="pseudo_mask", type=str, help="pesudo mask path")
    parser.add_argument("--train_list", default="configs/voc12/train_aug.txt", type=str)
    parser.add_argument("--weights", default='checkpoints/res38_cls.pth', type=str)
    parser.add_argument("--voc12_root", default='datasets/VOCdevkit/VOC2012/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument('--infer_list', default='configs/voc12/train_aug.txt', 
                        type=str, help='image list path')
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
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--low_alpha", default=1, type=float)
    parser.add_argument("--high_alpha", default=3, type=float)
    parser.add_argument("--radius", default=5, type=int)
        
    # hyper parameters settings
    parser.add_argument("--beta", default=11, type=int)
    parser.add_argument("--logt", default=7, type=int)
    parser.add_argument("--threshold", default=0.41, type=float, help='the optimal one obtained for seeds')
    args = parser.parse_args()
    return args


class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, imgarr):
        imgarr = (imgarr / 255. - self.mean) / self.std
        return imgarr.astype(np.float32)


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


def put_palette(seg_label, out_name):
    out = seg_label.astype(np.uint8)
    out = Image.fromarray(out, mode='P')
    # out.putpalette(data_voc.palette)
    out.save(out_name)
    
    
def build_train_dataset(args):
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
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            np.asarray,
            Normalize(),
            imutils.HWC_to_CHW],
        label_transform_list=[
            None,
            None,
            None,
            imutils.AvgPool2d(8)])
        
    return train_dataset


def build_train_dataloader(train_dataset, args):
    sampler_train = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=args.batch_per_gpu, 
        num_workers=args.num_workers,
        pin_memory=True, 
        drop_last=True)
    return sampler_train, train_data_loader
    
    
def build_infer_dataset(args):
    infer_dataset = VOC12ImageDataset(
        args.infer_list, 
        voc12_root=args.voc12_root,
        transform=torchvision.transforms.Compose(
            [np.asarray,
             Normalize(),
             imutils.HWC_to_CHW]))
    return infer_dataset # name, img-> C x H x W


def build_infer_dataloader(args, infer_dataset):
    infer_data_loader = DataLoader(
        infer_dataset, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=False) 
    return infer_data_loader
    

def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(
        databin, 
        shuffle=False, 
        num_workers=args.num_workers // n_gpus, 
        pin_memory=False) 
    
    stride = 8
    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        
        for iter_, (name, img) in enumerate(
                tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            name = name[0]
            original_shape = img.shape
            padded_size = (int(np.ceil(img.shape[2] / stride) * stride), int(np.ceil(img.shape[3] / stride) * stride))
            p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
            img = F.pad(img, p2d)

            dheight = int(np.ceil(img.shape[2] / stride))
            dwidth = int(np.ceil(img.shape[3] / stride))

            cam = np.load(osp.join(args.cam_out_dir, name + '.npy'), allow_pickle=True).item()

            cam_full_arr = np.zeros((args.num_classes, original_shape[2], original_shape[3]), np.float32)
            for k, v in cam.items():
                cam_full_arr[k + 1] = v

            cam_full_arr[0] = args.threshold
            cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

            aff_mat = torch.pow(model.forward(img.cuda(), True), args.beta)
            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            
            for _ in range(args.logt):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, stride, stride)

            cam_vec = cam_full_arr.view(args.num_classes, -1)
            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, args.num_classes, dheight, dwidth)

            cam_rw = F.interpolate(
                input=cam_rw,
                size=(img.shape[2], img.shape[3]), 
                mode='bilinear', 
                align_corners=False)
            _, cam_rw_pred = torch.max(cam_rw, 1)
                
            res = np.uint8(cam_rw_pred.cpu().data[0])[:original_shape[2], :original_shape[3]]
            
            # put_palette(res, osp.join(args.seg_out_dir, name + '.png'))
            mask = Image.fromarray(res, mode='P')
            mask.save(osp.join(args.seg_out_dir, name + '.png'))
            
        
def train_affinity(args):
    init_distributed_mode(args)
    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)
    same_seeds(args.seed)
    
    args.cam_out_dir = osp.join(args.work_space, args.cam_out_dir) 

    pyutils.Logger(osp.join(args.work_space, 'res38_aff.log'))
    
    train_dataset = build_train_dataset(args)
    sampler_train, train_data_loader = build_train_dataloader(train_dataset, args)
    
    args.world_size = dist.get_world_size()
    args.batch_size = args.batch_per_gpu * args.world_size
    max_step = len(train_dataset) * args.epoch // args.batch_size 
    args.max_step = max_step
    
    args_dict = vars(args)
    if dist.get_rank() == 0:
        pprint.pprint(args_dict)
    
    model = ResNet38d_Aff()
    weights_dict = torch.load(args.weights)
    
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolySGD([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    model.load_state_dict(weights_dict, strict=False)
    model.to(device)
    
    if args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.parallel.DistributedDataParallel(
        model, 
        find_unused_parameters=True, 
        device_ids=[args.local_rank])
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'bg_loss', 'fg_loss', 'neg_loss',
                                     'bg_cnt', 'fg_cnt', 'neg_cnt')
    timer = pyutils.Timer("Session started: ")
    eps = 1e-5
    
    for epoch in range(args.epoch):
        sampler_train.set_epoch(epoch)
        for iteration, pack in enumerate(train_data_loader):
            aff = model.forward(pack[0].cuda(non_blocking=True)) # B x 3 x H x W
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
        torch.save(model.module.state_dict(), 
                   osp.join(args.work_space, 'res38_aff_final.pth'))
    
    
def infer_affinity(args):
    args.cam_out_dir = osp.join(args.work_space, args.cam_out_dir)
    args.seg_out_dir = osp.join(args.work_space, args.seg_out_dir)
    data_mkdir(args.seg_out_dir)
    args.num_classes += 1 # add background
    pprint.pprint(vars(args))

    model = ResNet38d_Aff()

    model_dict = torch.load(
        osp.join(args.work_space, 'res38_aff_final.pth'), 
        map_location='cpu')
    model.load_state_dict(model_dict)
    model.eval()
    model.cuda()

    n_gpus = torch.cuda.device_count()
    
    dataset = build_infer_dataset(args)
    dataset = split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, 
                          args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()

    
if __name__ == '__main__':
    args = get_args_parser()
    
    if args.train:
        train_affinity(args)
        
    if args.inference:
        infer_affinity(args)
    