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
import warnings
warnings.filterwarnings("ignore")

from misc import torchutils, imutils
from utils import create_cam_model, parse_scales
from models.adapter_modules import resize_input_minbound


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


def get_args_parser():
    parser = argparse.ArgumentParser('Generating attention maps', add_help=False)
    # Model parameters
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--model', default='deit_small_mctgformer', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--checkpoint', default='', help='checkpoint for generating maps')
    parser.add_argument('--input_size', default=448, type=int, help='images input size')
    parser.add_argument('--min_size', default=448, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument('--dataset', default='', type=str, help='name of dataset')
    parser.add_argument('--work_space', default='results_voc/your_model', type=str, help='work space')
    parser.add_argument('--voc12_root', default='data/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument("--coco_root", default='data/MSCOCO', type=str, help="Path to MSCOCO")
    parser.add_argument("--train_list", default="train_aug_id.txt", type=str, 
                        help='train_id.txt or train_aug_id.txt')
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',help='')
    parser.set_defaults(pin_mem=True)

    # generating attention maps
    parser.add_argument('--layer-index', type=int, default=3, 
                        help='extract attention maps from the last layers')
    parser.add_argument("--scales", type=parse_scales, default=(1.0, ),
                        help="Multi-scale inferences")
    parser.add_argument("--cam_out_dir", default="cam", type=str)
    
    args = parser.parse_args()
    return args
                                                                                                                        

def normalize_cam(cam_mask):
    """Normalize the CAM mask for a single CAM."""
    # Find min and max values for each channel
    k = cam_mask.size(0)
    min_val = cam_mask.view(k, -1).min(dim=-1, keepdim=True)[0].view(k, 1, 1)
    max_val = cam_mask.view(k, -1).max(dim=-1, keepdim=True)[0].view(k, 1, 1)
    # Normalize each channel
    normalized_cam = (cam_mask - min_val) / (max_val - min_val + 1e-8)
    return normalized_cam


def flip_cam(cam_list):
    """Flip cam with scales in the given cam_list."""
    for i, cam_scale in enumerate(cam_list):
        group1, group2 = cam_scale[0], cam_scale[1]
        group2_flipped = torch.flip(group2, dims=[2])
        cam_list[i] = torch.stack([group1, group2_flipped])  
    cam_list = [torch.sum(cam, dim=0) for cam in cam_list]
    return cam_list


def simple_resize_test(inputs_flip, model, args):
    inputs = resize_input_minbound(
        inputs_flip.cuda(non_blocking=True),
        min_size=args.min_size)
    output_cam = model(inputs)
    return output_cam


def combine_images(results, h_splits, w_splits):
    batch_size, channels, _, _ = results[0].shape
    h = sum(h_splits)
    w = sum(w_splits)
    
    combined = torch.zeros((batch_size, channels, h, w), device=results[0].device)
    
    h_start = 0
    idx = 0
    for h_part in h_splits:
        w_start = 0
        for w_part in w_splits:
            combined[:, :, h_start:h_start + h_part, w_start:w_start + w_part] = results[idx]
            w_start += w_part
            idx += 1
        h_start += h_part
    return combined


def split_image_test(inputs, model, args):
    _, _, h, w = inputs.shape
    def split_length(length):
        if length > args.min_size:
            base_length = length // 3
            parts = [base_length] * 2  
            parts.append(length - sum(parts))
            return parts
        else:
            return [length]

    h_splits = split_length(h)
    w_splits = split_length(w)
    cropped_outputs = []
    
    h_start = 0
    for h_part in h_splits:
        w_start = 0
        for w_part in w_splits:
            cropped = inputs[:, :, h_start:h_start + h_part, w_start:w_start + w_part]
            crop_size = cropped.shape[2:]
            cropped_result = model(cropped)
            cropped_result = F.interpolate(
                cropped_result,
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            cropped_outputs.append(cropped_result)
            w_start += w_part
        h_start += h_part
        
    combined = combine_images(cropped_outputs, h_splits, w_splits)

    return combined


def multi_scale_test(model, images, args):
    output_cam_list = []
    for image in images:
        inputs_flip = image[0]
        _h, _w = inputs_flip.shape[2:]
        if max(_h, _w) <= args.min_size:
            inputs = resize_input_minbound(
                x=inputs_flip.cuda(non_blocking=True),
                min_size=args.min_size)
            cam = model(inputs)
        else:
            inputs = resize_input_minbound(
                x=inputs_flip.cuda(non_blocking=True),
                min_size=args.min_size)
            cam = split_image_test(inputs, model, args=args)
        output_cam_list.append(cam)
        
    return output_cam_list


def _work_trainset_psa(process_id, model, dataset, args):
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
            try:
                outputs = []
                for img in pack['img']:
                    out = model(resize_input_minbound(
                            x=img[0].cuda(non_blocking=True),
                            min_size=args.min_size)) # img[0]->[(2, 3, H', W')]
                    outputs.append(out)
                 # outputs->list[(2, n_cls, H/16, W/16)]
            except RuntimeError as e:
                if "out of memory" in str(e):
                    outputs = [model(resize_input_minbound(
                        x=img[0].cuda(non_blocking=True),
                        min_size=int(args.min_size * 0.5))) # img[0]->[(2, 3, H', W')]
                            for img in pack['img']] # outputs->list[(2, n_cls, H/16, W/16)]
                else:
                    raise e
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
                print(f"{(5*iter_+1) // (len(databin) // 20)} ", end='')


def _work_trainset_irn(process_id, model, dataset, args):
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
            strided_size = imutils.get_strided_size(size, 4)# floor(W'/4 x H'/4)
            
            if valid_cat.shape[0] == 0: # No validate category
                np.save(osp.join(args.cam_out_dir, img_name + '.npy'), dict())
                continue
            try:
                outputs = []
                for img in pack['img']:
                    out = model(resize_input_minbound(
                            x=img[0].cuda(non_blocking=True),
                            min_size=args.min_size)) # img[0]->[(2, 3, H', W')]
                    outputs.append(out)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    out = model(resize_input_minbound(
                            x=img[0].cuda(non_blocking=True),
                            min_size=int(0.5 * args.min_size))) # img[0]->[(2, 3, H', W')]
                    outputs.append(out)
                else:
                    raise e
            #========== strided cam list ====================================================#
            strided_cam_list = [# upsample all multi-scale CAMs to strided_size: (W'/4 x H'/4)
                F.interpolate(cam, strided_size, mode='bilinear', align_corners=False)
                for cam in outputs]
            strided_cam_list = flip_cam(strided_cam_list)
            strided_cam = torch.sum(torch.stack(strided_cam_list), 0) # (20, W'/4, H'/4)
            #======== high resolution cam list ===============================================#
            highres_cam_list = [# upsample all multi-scale CAMs to strided_up_size->floor(W^, H^)
                F.interpolate(cam, size, mode='bilinear', align_corners=False)
                for cam in outputs] # ->[(2, 20, W, H)
            highres_cam_list = flip_cam(highres_cam_list)
            highres_cam = torch.sum(torch.stack(highres_cam_list, 0), 0) # (20, W, H)

            strided_cam = strided_cam[valid_cat]
            strided_cam = normalize_cam(strided_cam)
            
            highres_cam = highres_cam[valid_cat]
            highres_cam = normalize_cam(highres_cam)
            
            cam_dict = {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()}
            np.save(osp.join(f"{args.cam_out_dir}_irn", img_name + '.npy'), cam_dict) 
            
            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def _work_testset(process_id, model, dataset, args):
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
            size = pack['size']        # image size->Torch.tensor [2]
            cls_labels, patch_labels, outputs = [], [], []
            
            # if os.path.exists(osp.join(args.cam_out_dir, img_name + '.npy')):
            #     continue
            try:
                for img in pack['img']:
                    cls_label, patch_label, output_cam = model.forward_with_label(
                        resize_input_minbound(
                            x=img[0].cuda(non_blocking=True),
                            min_size=args.min_size),
                        ) # img[0]->[(2, 3, H', W')]
                    cls_labels.append(cls_label)
                    patch_labels.append(patch_label)
                    outputs.append(output_cam)
                    
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    for img in pack['img']:
                        cls_label, patch_label, output_cam = model.forward_with_label(
                            resize_input_minbound(
                                x=img[0].cuda(non_blocking=True), 
                                min_size=int(args.min_size * 0.5)),
                            ) # img[0]->[(2, 3, H', W')]
                        cls_labels.append(cls_label)
                        patch_labels.append(patch_label)
                        outputs.append(output_cam)
                else:
                    raise e
            # print(f'{img_name}, {pseudo_label[0][0]} {patch_label[0][0]}')
            
            cls_label = cls_labels[0]     # choose the first scale
            patch_label = patch_labels[0] # choose the first scale
            result_label = cls_label[0] * cls_label[1] * patch_label[0] * patch_label[1] # combine flip results
            # pseudo_labels = cls_labels + patch_labels
            # result_label = pseudo_labels[0]
            # # for all rest tensors we do 'and' operation
            # for label in pseudo_labels[1:]:
            #     result_label = torch.bitwise_and(result_label, label)
            valid_cat = torch.nonzero(result_label)[:, 0]
            
            if valid_cat.shape[0] == 0: # No validate category
                np.save(osp.join(args.cam_out_dir, img_name + '.npy'), dict())
                continue
            #=================== high resolution cam list ===================#
            valid_cat = valid_cat.cpu() # because we use cuda before.
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
                print(f"{(5*iter_+1) // (len(databin) // 20)} ", end='')
                
                
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
        args.min_size = 448 if args.input_size >= 448 else 224
    else:
        raise NotImplementedError
    
    dataset, num_classes = build_dataset(
        is_train=False,
        make_cam=True,
        args=args)
    
    args.num_classes = num_classes
    
    model = create_cam_model(args)
    if 'model' in args.checkpoint:
        model_dict = args.checkpoint['model']
    else:
        model_dict = args.checkpoint
        
    model.load_state_dict(model_dict)
    model.eval()
    
    print(f'Using {args.checkpoint} for making cams.')
    n_gpus = torch.cuda.device_count()
    dataset = torchutils.split_dataset(dataset, n_gpus)
    
    cam_type = 'psa'
    if 'train' in args.train_list:
        if cam_type == 'psa':
            function = _work_trainset_psa
        else:
            function = _work_trainset_irn
    else: 
        function = _work_testset
        
    print(f'Using function {function}')
    
    print('[ ', end='')
    multiprocessing.spawn(
        function,
        nprocs=n_gpus,
        args=(model, dataset, args),
        join=True)
    print(']')

    torch.cuda.empty_cache()
