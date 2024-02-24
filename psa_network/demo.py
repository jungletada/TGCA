import os
import torch
import random
import argparse
import importlib
import numpy as np
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from tool import pyutils, imutils, torchutils
from network.transform import Normalize
from voc12.data_voc import VOC12AffDatasetM
cudnn.enabled = True


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="resnet38_aff", type=str)
    # ddp settings
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')  
    parser.add_argument('--gpu_id', default=0, type=int, help="which gpu to use")
    parser.add_argument("--local_rank", type=int, help='rank in current node')  
    parser.add_argument('--device', default='cuda',help='device id (i.e. 0 or 0,1 or cpu)')
    # training settings
    parser.add_argument("--batch_per_gpu", default=4, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--network", default="network.resnet38_aff", type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    # dataset settings
    parser.add_argument("--dataset", default="VOC", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--model_weights", default="checkpoints/res38_cls.pth", type=str)
    parser.add_argument("--cam_npy", default="results/MCTformerV2/voc/cam-npy", type=str)
    args = parser.parse_args()
    return args
        

def build_psa_dataset(model, args):
    """
    Input: dataset path and crf result path
    Return: image input->ndarray, crf label ->tuple
    """
    if args.dataset.lower().__contains__("voc"):
        train_dataset = VOC12AffDatasetM(
            img_name_list_path=args.train_list, 
            cam_npy_dir=args.cam_npy,
            voc12_root=args.dataset_root, 
            cropsize=args.crop_size, 
            low_alpha=1, 
            high_alpha=12,
            radius=5,
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
    
    elif args.dataset.lower().__contains__("coco"):
        train_dataset = None
    
    return train_dataset 


def build_dataloader(train_dataset, args):
    sampler_train = RandomSampler(train_dataset)
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
    pprint.pprint(args_dict)
        
        
if __name__ == '__main__':
    args = get_args_parser() 
    pyutils.Logger(os.path.join('checkpoints', args.session_name + '.log'))
    if args.dataset.lower().__contains__("voc"):
        args.dataset_root = "datasets/VOCdevkit/VOC2012"
        args.train_list = "psa/voc12/train_aug.txt"
    elif args.dataset.lower().__contains__("coco"):
        args.dataset_root = "datasets/MSCOCO"
        args.train_list = "psa/coco/train.txt"
        
    model = getattr(importlib.import_module(args.network), 'Net')()
    train_dataset = build_psa_dataset(model, args)
    sampler_train, train_data_loader = build_dataloader(train_dataset, args)
    torch.cuda.set_device(0)
    # print_info(args)
    
    # load model model_weights of resnet38d
    if args.model_weights[-7:] == '.params':
        import network.resnet38d
        assert args.network == "network.resnet38_aff"
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.model_weights)
    else:
        weights_dict = torch.load(args.model_weights)

    model.load_state_dict(weights_dict, strict=False)
    model.to("cuda")
    
    with torch.no_grad():
        for iteration, pack in enumerate(train_data_loader):
            aff = model.forward(pack[0].cuda())     # [4, 3, 448, 448]
            bg_label = pack[1][0].cuda(non_blocking=True) # [4, 34, 2496]
            fg_label = pack[1][1].cuda(non_blocking=True) # [4, 34, 2496]
            neg_label = pack[1][2].cuda(non_blocking=True)# [4, 34, 2496]
            print(f"{aff.shape}")# [4, 34, 2496]
            break
      
    