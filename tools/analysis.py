import os
import sys
import csv
import torch

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch.nn.functional as F
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from torch.backends import cudnn


cudnn.enabled = True
sys.path.append(".")
from misc import torchutils
from utils import create_cam_model
from models.adapter_modules import resize_input_minbound

import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


def get_args_parser():
    parser = argparse.ArgumentParser('Analyze attention maps', add_help=False)
    parser.add_argument("--task", default='sum_attention_weights', type=str)
    
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument('--work_space', default='results_voc/mcta', type=str, help='work space')
    parser.add_argument('--model', default='mcta', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--checkpoint', default='results_voc/mcta/mcta-deit-small-voc-7458.pth',
                        help='checkpoint for generating maps')
    parser.add_argument('--csv_path', default='attention_map_score.csv', type=str,
                        help='evaluation csv for cosine similarity.')
     # Model parameters
    parser.add_argument('--input_size', default=448, type=int, help='images input size')
    parser.add_argument('--min_size', default=448, type=int, help='images input size')
    
    # Dataset parameters
    parser.add_argument('--dataset', default='VOC12', type=str, help='name of dataset')
    parser.add_argument('--voc12_root', default='data/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument("--coco_root", default='data/MSCOCO', type=str, help="Path to MSCOCO")
    parser.add_argument("--train_list", default="data/VOCdevkit/VOC2012/ImageLists/train_id.txt", type=str, 
                        help='image name lists.')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',help='')
    parser.set_defaults(pin_mem=True)

    # generating attention maps
    parser.add_argument("--scales", default=(1.0,), help="Multi-scale inferences")
    parser.add_argument("--attn_dir", default="attns_dir", type=str)
    parser.add_argument("--log_dir", default="log_dir", type=str)
    return parser.parse_args()
                                                                                                                        
        
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


def draw_heat_map(attn_maps, args, img_name, task, n_dim=2):
    """Draw heat maps from attention maps.

    Args:
        attn_maps (Tensor): The attention maps to visualize.
        args (Namespace): The arguments containing output directory and other settings.
        img_name (str): The name of the image for saving the heat map.
        n_dim (int): The number of dimensions of the attention maps (2D or 3D).
    """
    
    cmap = "RdBu_r"
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
        output_path = os.path.join(args.attn_dir, f"{img_name}_{task}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.)
        plt.close()

    else:
        raise NotImplementedError


def calculate_cosine_similarity(outputs_block):
    """Calculate the cosine similarity for the outputs of each layer.

    Args:
        outputs_block (list): A list of outputs from the model layers.

    Returns:
        list: A list of mean cosine similarities for each layer.
    """
    cosine_similarities = []
    for layer_output in outputs_block: # pair-wise comparison: (N, C)
        patch_tokens = layer_output[0] # batch size=1
        patch_tokens = F.normalize(patch_tokens, p=2, dim=1, eps=1e-8)  # (N, C)
        cosine_sim = torch.matmul(patch_tokens, patch_tokens.T)
        all_pairs = cosine_sim.triu(diagonal=0)
        cosine_similarities.append(all_pairs.mean())

    return cosine_similarities


def calculate_loss_class_tokens(outputs_block, label):
    """Calculate the loss for class tokens based on the model outputs and labels.

    Args:
        outputs_block (list): A list of outputs from the model layers.
        label (Tensor): The ground truth labels for the inputs.
    """
    # loss = []
    # for layer, layer_output in enumerate(outputs_block):
    #     x_cls = layer_output[0] # B, K, C
    #     cls_logits = x_cls.mean(dim=-1)
    #     loss_per_layer = F.multilabel_soft_margin_loss(
    #         cls_logits, label)
    #     loss.append(loss_per_layer)
    # return loss
    final_out = outputs_block[-1][0]
    cls_logits = final_out.mean(dim=-1)
    print(cls_logits)
    print(label)


def calculate_class_score(all_cls_tokens, cls_label):
    f1_score_layer = []
    for cls_logits in all_cls_tokens: # pair-wise comparison: (N, C)
        cls_logits = cls_logits[0].mean(dim=-1)
        cls_pred = (cls_logits > 0).type(torch.int16)
        _f1_cls = metrics.f1_score(
            cls_label.cpu().numpy(),
            cls_pred.cpu().numpy())
        f1_score_layer.append(_f1_cls)
    return f1_score_layer

   
def _visualize_attention(model, dataset, args):
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    
    column_names = list(range(1, 12 + 1))

    with open(args.csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        with torch.no_grad():
            model.cuda()
            model.eval()
            for iter_, pack in enumerate(tqdm(data_loader)):
                img_name = pack['name'][0] # Img_id->str
                label = pack['label'][0]   # image-level label->Torch.Tensor [1]
                size = pack['size']        # image size->Torch.tensor [2]
                inputs = pack['img'][0]
                valid_cat = torch.nonzero(label)[:, 0] # get validate class->[#val_cls]

                # output_dict = model.forward(inputs[0].cuda(non_blocking=True))
                # all_cls_tokens = output_dict['all_cls']
                # all_patch_tokens = output_dict['all_patches']

                # results = calculate_cosine_similarity(outputs_block=all_cls_tokens)
                # results = calculate_class_score(all_cls_tokens, label.cuda())
                # rounded_values = [round(item.item(), 3) for item in results]
                # writer.writerow(rounded_values)
                task = 'cls_token'
                outputs = []
                for img in pack['img']:
                    inputs = img[0].cuda(non_blocking=True)
                    output = model.forward(inputs, return_type=task)
                    outputs.append(output)
                # attn_maps = outputs[0][:, 0, :, :].cpu().numpy()
                # attn_maps = np.sum(attn_maps, axis=0) # sum over all layer
                # draw_heat_map(attn_maps, args, img_name, task=task, n_dim=2)
                cls_token = F.relu(outputs[0][0, :, :].mean(dim=-1, keepdim=True))
                cls_x_cls = cls_token @ cls_token.T
                gt_cls_x_cls = label.unsqueeze(-1) @ label.unsqueeze(0)
                draw_heat_map(cls_x_cls.cpu().numpy(), args, img_name, task=task, n_dim=2)
                draw_heat_map(gt_cls_x_cls.cpu().numpy(), args, img_name, task=f'{task}_gt', n_dim=2)
                # Generate heatmaps for each layer's attention map
                if iter_ % (len(dataset) // 20) == 0:
                    print("%d " % ((5*iter_+1) // (len(dataset) // 20)), end='')


def analysis_per_layer_result(args):
    data_frame = pd.read_csv(args.csv_path, header=0)
    print(data_frame)
    column_means = data_frame.iloc[1:].astype(float).mean()
    for i in column_means:
        print(i)
 

def _eval_attention_scores(model, dataset, args):
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
            
    column_names = list(range(1, 12 + 1))
    nc = args.num_classes
        
    with open(args.csv_path, mode="w", newline="", encoding="utf-8") as file:
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

                if iter_ % (len(dataset) // 20) == 0:
                    print("%d" % ((5*iter_+1) // (len(dataset) // 20)), end='')
                    
        # Convert list of arrays to 2D numpy array
        cls_attn = np.array(all_cls_attn).mean(axis=0) # num_imges, L
        pat_attn = np.array(all_pat_attn).mean(axis=0) # num_imges, L

        writer.writerow(cls_attn)
        writer.writerow(pat_attn)


def _eval_attention_activation(model, dataset, args):
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
            
    column_names = list(range(1, 12 + 1))
    nc = args.num_classes
        
    with open(args.csv_path, mode="w", newline="", encoding="utf-8") as file:
        # writer = csv.writer(file)
        # writer.writerow(column_names)
        all_cls_attn = []
        all_pat_attn = []
        with torch.no_grad():
            model.cuda()
            model.eval()
            for iter_, pack in enumerate(tqdm(data_loader)):
                img_name = pack['name'][0] # Img_id->str
                label = pack['label'][0]   # image-level label->Torch.Tensor [1]
                size = pack['size']        # image size->Torch.tensor [2]
                
                idx_label = torch.nonzero(label, as_tuple=True)[0].tolist()
                outputs = [model(img[0].cuda(non_blocking=True), return_attn=True) # img[0]->[(2, 3, H', W')]
                                for img in pack['img']] # outputs->list[(2, n_cls, H/16, W/16)]
                attn_maps = outputs[0][:, 0, :, :] # L x (Cls+Np) x (Cls+Np), we choose non-flipped image.
                L, N, _ = attn_maps.shape
                x2cls, x2pat = attn_maps.split((nc, N-nc), dim=-1)
                
                #print(img_name, idx_cls, idx_label)
                if iter_ % (len(dataset) // 20) == 0:
                    print("%d" % ((5*iter_+1) // (len(dataset) // 20)), end='')
                    
        # # Convert list of arrays to 2D numpy array
        # cls_attn = np.array(all_cls_attn).mean(axis=0) # num_imges, L
        # pat_attn = np.array(all_pat_attn).mean(axis=0) # num_imges, L

        # writer.writerow(cls_attn)
        # writer.writerow(pat_attn)
        
        
def get_sum_of_atttention_weights(attn_maps, nc):
        """
        attn_maps: (L, N, N), where L: number of layers, N: (number of classes + number of patches)
        
        """
        all_cls_attn = []
        all_pat_attn = []
        L, N, _ = attn_maps.shape
        x2cls, x2pat = attn_maps.split((nc, N-nc), dim=-1)

        sum_cls = x2cls.sum(dim=-1).mean(dim=-1).cpu().numpy()
        sum_pat = x2pat.sum(dim=-1).mean(dim=-1).cpu().numpy()
        all_cls_attn.append(sum_cls)
        all_pat_attn.append(sum_pat)
        return all_cls_attn, all_pat_attn
    

if __name__ == '__main__':
    args = get_args_parser()
    args.attn_dir = os.path.join(args.work_space, args.attn_dir)
    args.csv_path = os.path.join(args.work_space, args.csv_path)
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
    model_dict = torch.load(args.checkpoint, map_location='cpu')['model']
    model.load_state_dict(model_dict)
    model.eval()

    print(f'Using {args.checkpoint} for analysis.')

    print('[ ', end='')
    _visualize_attention(model, dataset, args)
    print(']')
    # _eval_attention_activation(model, dataset, args)
    
    torch.cuda.empty_cache()
