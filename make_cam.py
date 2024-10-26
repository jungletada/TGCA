import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch.nn.functional as F
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
from torch.backends import cudnn
cudnn.enabled = True

from misc import torchutils
from net.modules import auto_resize_input
from utils import create_cam_model
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
    parser.add_argument("--cam_out_dir", default="cam_mask", type=str)
    
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


def sliding_window_test(x, model, patch_size, stride):
    """
    Perform a sliding window test with overlapping patches for a batch of images.

    Parameters:
    x (torch.Tensor): Input image tensor of shape (b, 3, H, W).
    model (torch.nn.Module): Model to process each patch.
    patch_size (int): Size of each patch (patch will be of shape (3, patch_size, patch_size)).
    stride (int): Stride for sliding window; must be less than patch_size for overlap.

    Returns:
    torch.Tensor: Reconstructed image tensor of shape (b, K, H, W), where K is the model's output channels.
    """
    x = auto_resize_input(x, min_size=patch_size)
    b, _, H, W = x.shape
    output_channels = model(x[:, :, :patch_size, :patch_size]).shape[1]
    # Calculate padding needed to make sure the last patch in each dimension is patch_size
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size

    # Apply padding to the input tensor
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)  # Shape: (b, 3, H + pad_h, W + pad_w)
    H_padded, W_padded = H + pad_h, W + pad_w

    # Calculate output dimensions after downsampling
    out_H, out_W = H_padded // 16, W_padded // 16

    # Prepare output and count tensors for averaging overlapping regions
    output = torch.zeros((b, output_channels, out_H, out_W), device=x.device)
    count = torch.zeros((b, output_channels, out_H, out_W), device=x.device)

    # Slide over the padded image
    for i in range(0, H_padded - patch_size + 1, stride):
        for j in range(0, W_padded - patch_size + 1, stride):
            # Extract patch for each image in the batch
            patch = x_padded[:, :, i:i + patch_size, j:j + patch_size]  # Shape: (b, 3, patch_size, patch_size)
            output_patch = model(patch)  # Shape: (b, K, patch_size//16, patch_size//16)

            # Calculate the coordinates in the output (downsampled by 16)
            out_i, out_j = i // 16, j // 16
            output[:, :, out_i:out_i + output_patch.shape[2], out_j:out_j + output_patch.shape[3]] += output_patch
            count[:, :, out_i:out_i + output_patch.shape[2], out_j:out_j + output_patch.shape[3]] += 1

    # Average overlapping areas
    output /= count

    # Crop to the original output size (H//16, W//16) if any padding was added
    return output[:, :, :H // 16, :W // 16]

    
def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    
    data_loader = DataLoader(
        databin, 
        shuffle=False, 
        num_workers=args.num_workers // n_gpus, 
        pin_memory=True)

    with torch.no_grad(), cuda.device(process_id):
        
        model.cuda()
        model.eval()
        
        for iter_, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            img_name = pack['name'][0] # Img_id->str
            label = pack['label'][0]   # image-level label->Torch.Tensor [1]
            size = pack['size']        # image size->Torch.tensor [2]
            
            valid_cat = torch.nonzero(label)[:, 0] # get validate class->[#val_cls]
            
            if valid_cat.shape[0] == 0: # No validate category
                np.save(osp.join(args.cam_out_dir, img_name + '.npy'), dict())
                continue

            outputs = [model.forward(img[0].cuda(non_blocking=True)) # img[0]->[(2, 3, H', W')]
                        for img in pack['img']] # outputs->list[(2, n_cls, H/16, W/16)]
            # outputs = []
            # for img in pack['img']:
            #     x = img[0].cuda(non_blocking=True)
            #     out = sliding_window_test(x, model, patch_size=448, stride=336)
            #     outputs.append(out)
            
            #=================== high resolution cam list ===================#
            upsample_cam_list = [# upsample all multi-scale CAMs
                    F.interpolate(cam, size, mode='bilinear', align_corners=False)
                    for cam in outputs] # ->[(2, Cls, H, W)
            upsample_cam_list = flip_cam(upsample_cam_list)
            upsample_cam = torch.sum(torch.stack(upsample_cam_list, 0), 0) # (Cls, H, W)
            
            upsample_cam = upsample_cam[valid_cat]
            upsample_cam = normalize_cam(upsample_cam)
            
            cam_dict = {}
            upsample_cam = upsample_cam.cpu().numpy()
            for i, cls in enumerate(valid_cat):
                cam_dict[cls] = upsample_cam[i]
                
            np.save(osp.join(args.cam_out_dir, img_name + '.npy'), cam_dict)  
        
            if process_id == n_gpus - 1 and iter_ % (len(databin) // 20) == 0:
                print("%d " % ((5*iter_+1) // (len(databin) // 20)), end='')
                    

if __name__ == '__main__':
    args = get_args_parser()
    args.cam_out_dir = os.path.join(args.work_space, args.cam_out_dir) 
    os.makedirs(args.cam_out_dir, exist_ok=True)

    from datasets_cam import build_dataset
    # change to multi-scale dataset
    if args.dataset == 'VOC12':
        args.dataset = 'VOC12MS'
        args.min_size = 448 if args.input_size >= 448 else 224
    elif args.dataset == 'COCO':
        args.dataset = 'COCOMS'
        args.min_size = 384 if args.input_size >= 384 else 224
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
    
    print(f'Using {args.checkpoint} for making cams.')
    n_gpus = torch.cuda.device_count()
    dataset = torchutils.split_dataset(dataset, n_gpus)
    
    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
