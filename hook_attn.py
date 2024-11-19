import os
import csv
import torch

import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from torch.backends import cudnn


cudnn.enabled = True

from misc import torchutils
from utils import create_cam_model
from net.msg_modules import resize_input_minbound

import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


def get_args_parser():
    parser = argparse.ArgumentParser('Generating attention maps', add_help=False)
    # Model parameters
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument('--model', default='deit_small_mctgformer', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--checkpoint', default='', help='checkpoint for generating maps')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--min_size', default=224, type=int, help='images input size')
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
                        help='configs/coco/train_id.txt or configs/voc12/train_aug_id.txt')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',help='')
    parser.set_defaults(pin_mem=True)

    # generating attention maps
    parser.add_argument('--layer-index', type=int, default=3, help='extract attention maps from the last layers')
    parser.add_argument("--scales", default=(1.0,), help="Multi-scale inferences")
    parser.add_argument("--attn_dir", default="attns_dir", type=str)
    parser.add_argument("--log_dir", default="log_dir", type=str)
    args = parser.parse_args()
    return args
                                                                                                                        
        
def normalize_cam(cam_mask):
    """Normalize the CAM mask."""
    for i in range(cam_mask.size(0)):
        channel = cam_mask[i]
        min_val = torch.min(channel)
        max_val = torch.max(channel)
        cam_mask[i] = (channel - min_val) / (max_val - min_val + 1e-8)
    
    return cam_mask


def flip_cam(cam_list):
    """Flip cam with scales in the given cam_list."""
    for i, cam_scale in enumerate(cam_list):
        group1, group2 = cam_scale[0], cam_scale[1]
        group2_flipped = torch.flip(group2, dims=[2])
        cam_list[i] = torch.stack([group1, group2_flipped])  
    cam_list = [torch.sum(cam, dim=0) for cam in cam_list]
    return cam_list


def draw_heat_map(attn_maps, args, img_name, n_dim=2):
    # Generate and save heatmaps using seaborn
    cmap = "mako"
    if n_dim == 3:
        for layer in range(attn_maps.shape[0]):
            plt.figure(figsize=(20, 20))
            sns.heatmap(attn_maps[layer], cmap=cmap, cbar=False, square=True)
            plt.axis("off")
            output_path = os.path.join(args.attn_dir, f"{img_name}_L{layer + 1}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.)
            plt.close()

    elif n_dim == 2:
        plt.figure(figsize=(20, 20))
        sns.heatmap(attn_maps, cmap=cmap, cbar=False, square=True)
        plt.axis("off")
        output_path = os.path.join(args.attn_dir, f"{img_name}_attnsum.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.)
        plt.close()
    
    else:
        raise NotImplementedError
    

def calculate_cosine_similarity(outputs_block):
    cosine_similarities = []
    for layer_output in outputs_block:
        # pair-wise comparison: (N, C)
        patch_tokens = layer_output[0]
        # Normalize the vectors to unit vectors to calculate cosine similarity
        patch_tokens = F.normalize(patch_tokens, p=2, dim=1, eps=1e-8)  # (N, C)
        # Compute cosine similarity for all pairs by matrix multiplication
        # The result will be a (N, N) matrix
        cosine_sim = torch.matmul(patch_tokens, patch_tokens.T)
        # Append to the list for this layer
        all_pairs = cosine_sim.triu(diagonal=0)
        cosine_similarities.append(all_pairs.mean())
        
    return cosine_similarities


def _work(model, dataset, args):
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    
    outputs_block = []
    # Define the hook function
    def hook_fn(module, input, output):
        # Capture the output `x` (which is the `output` argument in the hook function)
        outputs_block.append(output[0])  # Assuming `output[0]` is `x` and `output[1]` is `weights_j`

    # Register hooks to each block in self.blocks
    for i in range(model.stages):
        for j in range(model.stage_indices[i], model.stage_indices[i+1]):
            model.blocks[j].register_forward_hook(hook_fn)
            
    column_names = list(range(1, 12 + 1))
    
    with open("output.csv", mode="w", newline="", encoding="utf-8") as file:
        
        writer = csv.writer(file)
        writer.writerow(column_names)
        
        with torch.no_grad():
            model.cuda()
            model.eval()
            for iter_, pack in enumerate(tqdm(data_loader)):
                img_name = pack['name'][0] # Img_id->str
                label = pack['label'][0]   # image-level label->Torch.Tensor [1]
                size = pack['size']        # image size->Torch.tensor [2]
                
                valid_cat = torch.nonzero(label)[:, 0] # get validate class->[#val_cls]
                outputs = [model(img[0].cuda(non_blocking=True),
                                    return_attn=True) # img[0]->[(2, 3, H', W')]
                        for img in pack['img']] # outputs->list[(2, n_cls, H/16, W/16)]
                
                cosine_similarities = calculate_cosine_similarity(outputs_block)
                row_values = [item.item() for item in cosine_similarities]  # 将每个 tensor 转换为数值
                writer.writerow(row_values)
                # attn_maps = outputs[0][:, 0, :, :].cpu().numpy()    
                # attn_maps = np.sum(attn_maps, axis=0) # sum over all layer
                # draw_heat_map(attn_maps, args, img_name, n_dim=3)
                # print(attn_maps.shape, size)
                # Generate heatmaps for each layer's attention map
                if iter_ % (len(dataset) // 20) == 0:
                    print("%d " % ((5*iter_+1) // (len(dataset) // 20)), end='')
                    

def _work_attn(model, dataset, args):
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
            
    column_names = list(range(1, 12 + 1))
    nc = args.num_classes

    with open("output.csv", mode="w", newline="", encoding="utf-8") as file:
        
        writer = csv.writer(file)
        writer.writerow(column_names)
        all_cls_attn = []
        all_pat_attn = []
        with torch.no_grad():
            model.cuda()
            model.eval()
            for iter_, pack in enumerate(tqdm(data_loader)):
                img_name = pack['name'][0] # Img_id->str
                label = pack['label'][0]   # image-level label->Torch.Tensor [1]
                size = pack['size']        # image size->Torch.tensor [2]
        
                outputs = [model(img[0].cuda(non_blocking=True), return_attn=True) # img[0]->[(2, 3, H', W')]
                                for img in pack['img']] # outputs->list[(2, n_cls, H/16, W/16)]
                attn_maps = outputs[0][:, 0, :, :] # L x (Cls+Np) x (Cls+Np)
                L, N, _ = attn_maps.shape
                x2cls, x2pat = attn_maps.split((nc, N-nc), dim=-1)

                sum_cls = x2cls.sum(dim=-1).mean(dim=-1).cpu().numpy()
                sum_pat = x2pat.sum(dim=-1).mean(dim=-1).cpu().numpy()
                all_cls_attn.append(sum_cls)
                all_pat_attn.append(sum_pat)
                # writer.writerow(sum_cls)
                # attn_maps = np.sum(attn_maps, axis=0) # sum over all layer
                # draw_heat_map(attn_maps, args, img_name, n_dim=3)
                # print(attn_maps.shape, size)
                # Generate heatmaps for each layer's attention map
                if iter_ % (len(dataset) // 20) == 0:
                    print("%d" % ((5*iter_+1) // (len(dataset) // 20)), end='')

        # Convert list of arrays to 2D numpy array
        cls_attn = np.array(all_cls_attn).mean(axis=0) # num_imges, L
        pat_attn = np.array(all_pat_attn).mean(axis=0) # num_imges, L

        writer.writerow(cls_attn)
        writer.writerow(pat_attn)


if __name__ == '__main__':
    args = get_args_parser()
    args.attn_dir = os.path.join(args.work_space, args.attn_dir) 
    os.makedirs(args.attn_dir, exist_ok=True)

    from datasets_cam import build_dataset
    # change to multi-scale dataset
    if args.dataset == 'VOC12':
        args.dataset = 'VOC12MS'
        args.num_classes = 20
    elif args.dataset == 'COCO':
        args.dataset = 'COCOMS'
        args.num_classes = 80
    else:
        raise NotImplementedError
    
    dataset, num_classes = build_dataset(
        is_train=False, make_cam=True, args=args)
    args.num_classes = num_classes
    
    model = create_cam_model(args)
    model_dict = torch.load(args.checkpoint,map_location='cpu')['model']
    model.load_state_dict(model_dict)
    model.eval()
    
    print(f'Using {args.checkpoint} for making cams.')
    
    print('[ ', end='')
    _work_attn(model, dataset, args)
    print(']')

    torch.cuda.empty_cache()
