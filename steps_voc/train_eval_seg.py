import cv2
import sys, os
import os.path as osp
import importlib
import argparse
import random
import numpy as np
from tqdm import tqdm
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


sys.path.append(osp.dirname(__file__) + os.sep + '../')
import logging
import utils 
from seg_tool import imutils,  pyutils, torchutils
from seg_tool.metrics import Evaluator
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


def str2bool(v):
    if v.lower() in ('yes','true','t','y','1','True'):
        return True
    elif v.lower() in ('no','false','f','n','0','False'):
        return False


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, type=str2bool, help='train model')  
    parser.add_argument('--evaluate', default=True, type=str2bool, help='evaluate model')  
    # ddp settings
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')  
    parser.add_argument('--gpu_id', default=0, type=int, help="which gpu to use")
    parser.add_argument("--local_rank", type=int, help='rank in current node')  
    parser.add_argument('--device', default='cuda',help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--num_workers', default=8, type=int)
    
    parser.add_argument("--work_space", default="results_voc/MCTG", type=str)
    parser.add_argument("--pred_path", default=None, type=str)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--batch_per_gpu", default=8, type=int)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--num_epochs", default=60, type=int)
    parser.add_argument("--warmup_step", default=1500, type=int)
    parser.add_argument("--init_weights", default="", type=str)

    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--wt_dec", default=1e-5, type=float)
    parser.add_argument("--model_name", default="resnet38_seg", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument('--print_intervals', type=int, default=50)
    
    parser.add_argument("--use_crf", default=False, type=str2bool)
    parser.add_argument("--scales", default=(1.0, ), help="Multi-scale inferences")
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
        
        
def train(args):
    init_distributed_mode(args)
    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)
    
    same_seeds(args.seed)
    pyutils.Logger(os.path.join(args.work_space, 'voc_segmentation.log'))
    args.ckpt_path = os.path.join(args.work_space, 'seg_ckpt')
    utils.data_mkdir(args.ckpt_path)
    
    dataset = VOCAugSegmentationDataset(
        root="datasets/VOCdevkit/VOC2012",
        split="train_aug",
        pseudo_dir=None,
        ignore_label=255,
        is_train=True,
        base_size=None,
        crop_size=321,
        scales=(0.7, 1.3))
    
    sampler = DistributedSampler(dataset)
    
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
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
    
    optimizer = torchutils.PolyOptimizerAdamW([
        {'params': model.get_1x_lr_params(), 'lr': args.lr},
        {'params': model.get_10x_lr_params(), 'lr': 10 * args.lr}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step, warmup_step=args.warmup_step,)

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

    for epoch in range(1, args.num_epochs + 1):
        for iteration, (name, images, labels) in enumerate(data_loader):
            labels = labels.to(device)
            images = images.to(device)
            
            pred = model(images)
            pred = F.interpolate(
                pred, 
                size=images.shape[2:], 
                mode='bilinear', 
                align_corners=False)
            
            loss = criterion(pred, labels)
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
                
        if dist.get_rank() == 0 and epoch % 10 == 0:
            torch.save(model.module.state_dict(), 
                    os.path.join(args.ckpt_path, args.model_name + f'_{epoch}.pth')) 
                 
        if dist.get_rank() == 0 and epoch == args.num_epochs:
            torch.save(model.module.state_dict(), 
                    os.path.join(args.ckpt_path, args.model_name + '_last.pth'))


def evaluate(args):
    session_name = "Segmentation Inference"
    log_path = os.path.join(args.work_space, 'eval_seg.log')
    args.ckpt_path = os.path.join(args.work_space, 'seg_ckpt')
    
    utils.logger_info(logger_name=session_name, log_path=log_path)
    logger = logging.getLogger(session_name)
    
    logger.info(f"Evaluation log path: {log_path}")
    logger.info(f"Multi-scale test: {tuple(args.scales)}, use CRF: {args.use_crf}")
   
    if args.pred_path is not None: 
        save_pred = True
        pred_path = os.path.join(args.work_space, args.pred_path)
        if args.use_crf:
            pred_path = pred_path + "_crf"
        utils.data_mkdir(pred_path)
        logger.info(f"Multi-scale Evaluation with save path: {pred_path}")
    else:
        save_pred = False
        logger.info("Not to save Multi-scale Evaluation results.")

    model = ResNet38d_Seg(num_classes=args.num_classes)
    ckpt = os.path.join(args.ckpt_path, "resnet38_seg_last.pth")
    model.load_state_dict(torch.load(ckpt), strict=True)
    seg_evaluator = Evaluator(num_class=args.num_classes)
    
    model.eval()
    model.cuda()
    
    dataset = VOCAugSegmentationDataset(
        root="datasets/VOCdevkit/VOC2012",
        split="val",
        pseudo_dir=None,
        ignore_label=255,
        is_train=False)
    
    data_loader = DataLoader(
        dataset,
        batch_size=1, 
        num_workers=args.num_workers,
        pin_memory=True, 
        drop_last=False)
    
    scales = tuple(args.scales)

    with torch.no_grad():
        for image_id, original_images, images_tensor, labels in tqdm(data_loader):       
            H, W = images_tensor.shape[2:]
            probs = torch.zeros((1, args.num_classes, H, W)).cuda()
            for s in scales:
                new_size = (int(H * s), int(W * s))
                img_scale = F.interpolate(
                    images_tensor, 
                    size=new_size, 
                    mode='bilinear', 
                    align_corners=True)
                prob = model(x=img_scale.cuda())
                prob = F.interpolate(
                    prob, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False)
                prob = F.softmax(prob, dim=1)
                probs = torch.max(probs, prob)

            output = probs.cpu().data[0].numpy()
            labels = np.asarray(labels.cpu().data[0].numpy())
            
            if args.use_crf:
                original_image = original_images.cpu().data[0].numpy()
                crf_output = imutils.crf_inference_inf(original_image, output, labels=args.num_classes)
                pred = np.argmax(crf_output, 0)
            else:
                pred = np.argmax(output, axis=0)

            seg_evaluator.add_batch(labels, pred)

            if save_pred:
                out = pred.astype(np.uint8)
                out = Image.fromarray(out, mode='P')
                out.putpalette(palette)
                save_name = os.path.join(pred_path, image_id[0] + '.png')
                out.save(save_name)

        IoU, mIoU = seg_evaluator.Mean_Intersection_over_Union()

        for k in range(args.num_classes):
            logger.info('class {:2d} {:12} IU {:.3f}'.format(k, classes[k], IoU[k]))
            
        logger.info('mIoU = {:.3f}'.format(mIoU))


if __name__ == '__main__':
    args = get_args_parser()
    
    if args.train:
        train(args)
        
    if args.evaluate:
        evaluate(args)