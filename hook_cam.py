import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import os.path as osp
import torch.nn.functional as F
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from torch.backends import cudnn

from misc import torchutils
from utils import create_cam_model
import warnings
warnings.filterwarnings("ignore")


 # Updated global dictionary to store both pre and post features
intermediate_features = {
    'cls2pat_pre': None,    # capture input to model.graph_b
    'cls2pat_post': None,   # capture output of model.graph_b
    'cls2spt_pre': None,    # capture input to model.graph_s
    'cls2spt_post': None,   # capture output of model.graph_s
}


def get_args_parser():
    parser = argparse.ArgumentParser('Generating attention maps', add_help=False)
    # Model parameters
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument('--model', default='deit_small_mctgformer', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--checkpoint', default='', help='checkpoint for generating maps')
    parser.add_argument('--input_size', default=448, type=int, help='images input size')
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
    parser.add_argument('--layer-index', type=int, default=3, help='extract attention maps from the last layers')
    parser.add_argument("--scales", default=(1.0,), help="Multi-scale inferences")
    parser.add_argument("--feat_out_dir", default="features", type=str)
    
    args = parser.parse_args()
    return args


def get_intermediate_feature(name, hook_type='post'):
    """hook function to capture both pre and post intermediate features"""
    def pre_hook(model, input):
        intermediate_features[name] = input[0].detach() # Captures input to the layer

    def post_hook(model, input, output):
        intermediate_features[name] = output.detach() # Captures output from the layer

    # Return the appropriate hook function based on the specified hook_type
    if hook_type == 'pre':
        return pre_hook
    elif hook_type == 'post':
        return post_hook
    else:
        raise ValueError("hook_type must be 'pre' or 'post'")


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    
    data_loader = DataLoader(
        databin, 
        shuffle=False, 
        num_workers=args.num_workers // n_gpus, 
        pin_memory=False)
    
    stage = 3
    assert stage in [0, 1, 2, 3]
    
    # Register pre and post hooks for model.graph_b
    model.spatial_fuse[stage].graph_b.register_forward_pre_hook(
        get_intermediate_feature('cls2pat_pre', hook_type='pre'))
    model.spatial_fuse[stage].graph_b.register_forward_hook(
        get_intermediate_feature('cls2pat_post', hook_type='post'))

    # Register pre and post hooks for model.graph_s
    model.spatial_fuse[stage].graph_s.register_forward_pre_hook(
        get_intermediate_feature('cls2spt_pre', hook_type='pre'))
    model.spatial_fuse[stage].graph_s.register_forward_hook(
        get_intermediate_feature('cls2spt_post', hook_type='post'))
    
    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        for iter_, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            img_name = pack['name'][0] 
            img = pack['img']
            outputs = model.forward(img.cuda(non_blocking=True))
            
            cls2pat_pre_feat = intermediate_features['cls2pat_pre']
            cls2pat_post_feat = intermediate_features['cls2pat_post']

            # Now you have access to cls2pat both before and after model.graph_b
            B, _, H, W = cls2pat_pre_feat.shape
            cls2pat_pre_feat = cls2pat_pre_feat.view(B, -1, args.num_classes, H, W)
            cls2pat_post_feat = cls2pat_post_feat.view(B, -1, args.num_classes, H, W)
            print("cls2pat pre-processed features shape:", cls2pat_pre_feat.shape)
            print("cls2pat post-processed features shape:", cls2pat_post_feat.shape)
            
            cls2spt_pre_feat = intermediate_features['cls2spt_pre']
            cls2spt_post_feat = intermediate_features['cls2spt_post']
            B, _, H, W = cls2spt_pre_feat.shape
            cls2spt_pre_feat = cls2spt_pre_feat.view(B, -1, args.num_classes, H, W)
            cls2spt_post_feat = cls2spt_post_feat.view(B, -1, args.num_classes, H, W)
            print("cls2spt pre-processed features shape:", cls2spt_pre_feat.shape)
            print("cls2spt post-processed features shape:", cls2spt_post_feat.shape)
            break
            
            if process_id == n_gpus - 1 and iter_ % (len(databin) // 20) == 0:
                print("%d " % ((5*iter_+1)//(len(databin) // 20)), end='')
                

if __name__ == '__main__':
    args = get_args_parser()
    args.feat_out_dir = os.path.join(args.work_space, args.feat_out_dir) 
    os.makedirs(args.feat_out_dir, exist_ok=True)
    
    from datasets_cam import build_dataset
    
    if args.dataset == 'VOC12':
        args.dataset = 'VOC12Img'
        
    dataset, args.num_classes = build_dataset(
        is_train=False, make_cam=True, args=args)
    
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