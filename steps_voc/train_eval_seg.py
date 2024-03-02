import cv2
import sys, os
import os.path as osp
import importlib
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from tool import imutils
from tool import pyutils, torchutils
from tool.metrics import Evaluator
import PIL.Image as Image

sys.path.append(osp.dirname(__file__) + os.sep + '../')
import logging
import utils 
from data.voc12.dataloader import VOCAugSegmentationDataset
from net.resnet38d_seg import ResNet38d_Seg
cudnn.enabled = True


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]


classes = np.array(('background',  # always index 0
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor'))


def crf_postprocess(pred_prob, ori_img, labels=21):
    crf_score = imutils.crf_inference_inf(ori_img, pred_prob, labels=labels)
    return crf_score


def str2bool(v):
    if v.lower() in ('yes','true','t','y','1','True'):
        return True
    elif v.lower() in ('no','false','f','n','0','False'):
        return False


def get_args_parser():
    parser = argparse.ArgumentParser()
    # ddp settings
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')  
    parser.add_argument('--gpu_id', default=0, type=int, help="which gpu to use")
    parser.add_argument("--local_rank", type=int, help='rank in current node')  
    parser.add_argument('--device', default='cuda',help='device id (i.e. 0 or 0,1 or cpu)')
    
    parser.add_argument("--work_space", default="results_voc/MCTG", type=str)
    parser.add_argument("--save_path", default=None, type=str)

    parser.add_argument('--seed', default=0, type=int)
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
    parser.add_argument("--use_crf", default=True, type=str2bool)
    parser.add_argument("--scales", type=float, nargs='+')
    args = parser.parse_args()
    return args 

  
def evaluate(args):
    utils.data_mkdir(args.work_space)
    session_name = "Segmentation Inference"
    log_path = os.path.join(args.work_space, 'eval_seg.log')
    utils.logger_info(logger_name=session_name, log_path=log_path)
    logger = logging.getLogger(session_name)
    
    logger.info(f"Evaluation log path: {log_path}")
    logger.info(f"Multi-scale test: {tuple(args.scales)}, use CRF: {args.use_crf}")
   
    if args.save_palette: 
        pred_save_path = os.path.join(args.work_space, 'val_ms_crf_color')
        utils.data_mkdir(pred_save_path)
        logger.info(f"Multi-scale Evaluation with CRF save path: {pred_save_path}")
    else:
        logger.info("No to save Multi-scale Evaluation results.")
        
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    model = getattr(importlib.import_module('network.resnet38_seg'), 'ResNet38d_Seg')(num_classes=args.num_classes)

    model.load_state_dict(torch.load(args.checkpoint))
    seg_evaluator = Evaluator(num_class=args.num_classes)
    
    model.eval()
    model.cuda()
    base_img_path = args.img_path
    name_list = open(utils.get_dataset_imglist(args.dataset)).readlines()
    
    dataset = VOCAugSegmentationDataset(
        root="datasets/VOCdevkit/VOC2012",
        split="val",
        augment=False,
        pseudo_dir=None,
        ignore_label=255,
        base_size=None,
        crop_size=321,
        scales=(0.7, 1.3),
        flip=True,
    )
    with torch.no_grad():
        for idx in tqdm(range(len(name_list))):
            name = name_list[idx]
            img_temp = cv2.imread(os.path.join(base_img_path, name.strip() + '.jpg'))
            img_original, img_tensor = transform_image(image=img_temp)
            N, C, H, W = img_tensor.size()
            probs = torch.zeros((N, args.num_classes, H, W)).cuda()
            
            if args.scales:
                scales = tuple(args.scales)
                
            for s in scales:
                new_hw = [int(H * s), int(W * s)]
                img_scale = F.interpolate(img_tensor, new_hw, mode='bilinear', align_corners=True)
                prob = model(x=img_scale)
                prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=False)
                prob = F.softmax(prob, dim=1)
                probs = torch.max(probs, prob)

            output = probs.cpu().data[0].numpy()

            if args.use_crf:
                crf_output = crf_postprocess(output, img_original)
                pred = np.argmax(crf_output, 0)
            else:
                pred = np.argmax(output, axis=0)

            gt = Image.open(os.path.join(args.gt_path, name.strip() + '.png'))
            gt = np.asarray(gt)
            seg_evaluator.add_batch(gt, pred)

            # save_path = os.path.join(args.save_path, i.strip() + '.png')
            # cv2.imwrite(save_path, pred.astype(np.uint8))

            if args.save_palette:
                out = pred.astype(np.uint8)
                out = Image.fromarray(out, mode='P')
                out.putpalette(palette)
                out_name = os.path.join(pred_save_path, name.strip() + '.png')
                out.save(out_name)

        IoU, mIoU = seg_evaluator.Mean_Intersection_over_Union()

        str_format = "{:<15s}\t{:<15.2%}"
        for k in range(args.num_classes):
            logger.info('class {:2d} {:12} IU {:.3f}'.format(k, classes[k], IoU[k]))
        logger.info('mIoU = {:.3f}'.format(mIoU))


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
        
        
def train(args):
    init_distributed_mode(args)
    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)
    
    same_seeds(args.seed)
    pyutils.Logger(os.path.join(args.work_space, 'voc_segmentation.log'))
    
    dataset = VOCAugSegmentationDataset(
        root="datasets/VOCdevkit/VOC2012",
        split="train_aug",
        pseudo_dir=None,
        ignore_label=255,
        augment=True,
        base_size=None,
        crop_size=321,
        scales=(0.7, 1.3),
        flip=True)
    sampler_train = DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=args.batch_per_gpu, 
        num_workers=args.num_workers,
        pin_memory=True, 
        drop_last=True)

    train_size = len(dataset)
    args.world_size = dist.get_world_size()
    args.batch_size = args.batch_per_gpu * args.world_size
    num_batches_per_epoch = train_size // args.batch_size
    args.max_step = args.num_epochs * num_batches_per_epoch

    model = ResNet38d_Seg(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.init_weights), strict=False)
    
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
    
    criterion = nn.CrossEntropyLoss(
        weight=None, ignore_index=255, reduction='mean').to(device)
    
    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")

    for epoch in range(args.num_epochs):
        for iteration, (name, image, label) in enumerate(data_loader):
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