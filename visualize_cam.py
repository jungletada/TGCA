import os
import cv2
import torch
import imageio
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn.functional as F
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from torch.backends import cudnn
cudnn.enabled = True
import matplotlib.pyplot as plt
from misc import torchutils
from utils import create_cam_model
import warnings
warnings.filterwarnings("ignore")


CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train','tvmonitor']


def get_args_parser():
    parser = argparse.ArgumentParser('Generating attention maps', add_help=False)
    # Model parameters
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument('--model', default='deit_small_mctgformer', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--checkpoint', default='', help='checkpoint for generating maps')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--dataset', default='', type=str, help='name of dataset')
    parser.add_argument('--work_space', default='results_voc/your_model', type=str, help='work space')
    parser.add_argument('--voc12_root', default='datasets/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument("--coco_root", default='datasets/MSCOCO', type=str, help="Path to MSCOCO")
    parser.add_argument("--train_list", default="configs/voc12/train_aug_id.txt", type=str, 
                        help='[configs/coco/train_id.txt] or [configs/voc12/train_aug_id.txt]')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',help='')
    parser.set_defaults(pin_mem=True)

    # generating attention maps
    parser.add_argument('--fuse_layer', type=int, default=3, help='extract attention maps from the last layers')
    parser.add_argument("--scales", default=(1.0, 0.75, 1.25), help="Multi-scale inferences")
    parser.add_argument("--vis_cam_dir", default="vis_cam", type=str)
    
    
    args = parser.parse_args()
    return args
                                                                                                                        
        
def normalize_cam(cam_mask):
    for i in range(cam_mask.size(0)):
        channel = cam_mask[i]
        min_val = torch.min(channel)
        max_val = torch.max(channel)
        cam_mask[i] = (channel - min_val) / (max_val - min_val + 1e-8)
    
    return cam_mask


def flip_cam(cam_list):
    for i in range(len(cam_list)):
        cam_scale = cam_list[i]
        group1, group2 = cam_scale[0], cam_scale[1]
        group2_flipped = torch.flip(group2, dims=[2])
        cam_list[i] = torch.stack([group1, group2_flipped])
        
    cam_list = [torch.sum(cam, dim=0) for cam in cam_list]
    return cam_list

     
def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    
    data_loader = DataLoader(
        databin, 
        shuffle=False, 
        num_workers=args.num_workers // n_gpus, 
        pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        for iter_, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            img_name = pack['name'][0] # Img_id->str
            label = pack['label'][0]   # image-level label->Torch.Tensor [1]
            size = pack['size']        # image size->Torch.tensor [2]
            original_img = pack['original']
            outputs = [model.forward(img[0].cuda(non_blocking=True)) # img[0]->[(2, 3, H', W')]
                       for img in pack['img']] # outputs->list[(2, n_cls, H/16, W/16)]
            
            #========================== high resolution cam list ==========================#
            upsample_cam_list = [ # upsample all multi-scale CAMs
                F.interpolate(cam, size, mode='bilinear', align_corners=False)
                for cam in outputs] # ->[(2, n_cls, H, W)
            upsample_cam_list = flip_cam(upsample_cam_list)
            upsample_cam = torch.sum(torch.stack(upsample_cam_list, 0), 0) # (20, H, W)
            
            valid_cat = torch.nonzero(label)[:, 0] # get validate class->[#val_cls]
            if valid_cat.shape[0] == 0: # No validate category
                np.save(osp.join(args.vis_cam_dir, img_name.replace('jpg', 'npy')),
                        dict())
                continue
            
            upsample_cam = upsample_cam[valid_cat]
            upsample_cam = normalize_cam(upsample_cam)
            
            cam_dict = {}
            upsample_cam = upsample_cam.cpu().numpy()
            
            for i, cls in enumerate(valid_cat):
                cam_dict[cls] = upsample_cam[i]
                img_cls_name = osp.join(args.vis_cam_dir, img_name + f'_{CAT_LIST[cls]}.jpg')
                # Convert heatmap value from [0, 1] to [0, 255]
                cam_normalized = cv2.normalize(cam_dict[cls], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # Apply Color Map
                heatmap = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)
                alpha = 0.5  # transparentcy
                base_img = (original_img[0]).cpu().numpy()
                overlay = cv2.addWeighted(base_img, 1 - alpha, heatmap, alpha, 0)
                cv2.imwrite(img_cls_name, overlay)
            
            if process_id == n_gpus - 1 and iter_ % (len(databin) // 20) == 0:
                print("%d " % ((5*iter_+1)//(len(databin) // 20)), end='')
                
          
if __name__ == '__main__':
    args = get_args_parser()
    args.vis_cam_dir = os.path.join(args.work_space, args.vis_cam_dir) 
    os.makedirs(args.vis_cam_dir, exist_ok=True)
    
    from datasets_cam import build_dataset
    # change to multi-scale dataset
    if args.dataset == 'VOC12':
        args.dataset = 'VOC12MS' 
    elif args.dataset == 'COCO':
        args.dataset = 'COCOMS'
    else:
        raise NotImplementedError
    
    dataset, num_classes = build_dataset(
        is_train=False, make_cam=True, args=args)
    args.num_classes = num_classes
    
    model = create_cam_model(args)
    model_dict = torch.load(
        args.checkpoint, 
        map_location='cpu')['model']
    
    model.load_state_dict(model_dict)
    model.eval()

    n_gpus = torch.cuda.device_count()
    dataset = torchutils.split_dataset(dataset, n_gpus)
    
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
   
    torch.cuda.empty_cache()