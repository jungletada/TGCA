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
            # try:
            #     outputs = [model.forward(img[0].cuda(non_blocking=True)) # img[0]->[(2, 3, H', W')]
            #             for img in pack['img']] # outputs->list[(2, n_cls, H/16, W/16)]
            # except:
            #     with open('error.txt', 'a') as f:
            #         f.write(img_name + '\n')
            #     continue
            # if os.path.exists((osp.join(args.cam_out_dir, img_name + '.npy'))):
            #     continue
            
            # if size[0] <= args.input_size and size[1] <= args.input_size:
            #     outputs = [model.forward(img[0].cuda(non_blocking=True)) # img[0]->[(2, 3, H', W')]
            #             for img in pack['img']] # outputs->list[(2, n_cls, H/16, W/16)]
            # else: # use sliding window test
            #     outputs = []
            #     for img in pack['img']:
            #         input_img = img[0].cuda(non_blocking=True)
            #         out = resize_and_sliding_window_test(
            #             input_img, 
            #             model, 
            #             train_size=args.input_size,
            #             stride=args.input_size // 4)
            #         outputs.append(out)
            
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
                    
 
def sliding_window_tensor(tensor, window_size, stride):
    """Splits the input tensor using a sliding window of specified size and stride."""
    patches = tensor.unfold(2, window_size, stride).unfold(3, window_size, stride)
    patches = patches.contiguous().view(-1, tensor.size(1), window_size, window_size)
    return patches


def combine_patches(patches, original_shape, stride):
    """Recombine patches into the original image shape."""
    n, c, h, w = original_shape
    patch_size = patches.size(2)
    output = torch.zeros(n, c, h, w, dtype=patches.dtype, device=patches.device)
    # Overlap counter to average overlapping regions
    count = torch.zeros_like(output)
    patch_idx = 0
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            output[:, :, i:i+patch_size, j:j+patch_size] += patches[patch_idx]
            count[:, :, i:i+patch_size, j:j+patch_size] += 1
            patch_idx += 1

    # Avoid division by zero for regions that are not overlapped
    output = output / torch.clamp(count, min=1)
    return output


def resize_and_sliding_window_test(X, model, train_size=448, stride=224):
    batch_size, _, height, width = X.shape
    
    # Rescale the tensor so that the shortest side is at least train_size
    min_side = min(height, width)
    scale_factor = train_size / min_side
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Rescale the input tensor
    X_rescaled = F.interpolate(X, size=(new_height, new_width), mode='bilinear', align_corners=False)
    
    # Split the rescaled tensor into patches of size (2, 3, 448, 448)
    patches = sliding_window_tensor(X_rescaled, train_size, stride)
    # Pass each patch through the model and get the output
    outputs = []
    for patch in patches:
        patch = patch.unsqueeze(0)  # Add batch dimension
        output = model(patch)  # Forward pass through the model
        # Rescale the output back to the patch size
        output_rescaled = F.interpolate(output, size=(train_size, train_size), mode='bilinear', align_corners=False)
        outputs.append(output_rescaled.squeeze(0))  # Remove batch dimension

    # Concatenate all the outputs
    outputs = torch.stack(outputs, dim=0)
    channels = outputs.shape[1]
    # Combine the model's outputs to get the combined result
    output_shape = (batch_size, channels, new_height, new_width)
    combined_output = combine_patches(outputs, output_shape, stride)
    
    # Resize the final combined output to the original input size
    # final_output = F.interpolate(combined_output, size=(height, width), mode='bilinear', align_corners=False)
    
    return combined_output #final_output

       
if __name__ == '__main__':
   
    args = get_args_parser()
    args.cam_out_dir = os.path.join(args.work_space, args.cam_out_dir) 
    os.makedirs(args.cam_out_dir, exist_ok=True)
    
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
    
    print(f'Using {args.checkpoint} for making cams.')
    n_gpus = torch.cuda.device_count()
    dataset = torchutils.split_dataset(dataset, n_gpus)
    
    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
