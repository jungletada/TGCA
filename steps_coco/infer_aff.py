import os
import sys
import torch
import torchvision
import numpy as np

import os.path as osp
import argparse
import importlib
from tqdm import tqdm

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import multiprocessing, cuda
import torch.nn.functional as F
import PIL.Image as Image
from pathlib import Path

sys.path.append(osp.dirname(__file__) + os.sep + '../')
from psa_network.tool import pyutils, imutils
from misc import torchutils
from psa_network.network.transform import Normalize
from psa_network.network.resnet38_aff import Net
from data.coco.dataloader_psa import COCOImageDataset
cudnn.enabled = True


def get_args_parser():
    parser = argparse.ArgumentParser()
    # checkpoint and data settings
    parser.add_argument("--work_space", default="results_coco/MCTG", type=str)
    parser.add_argument("--coco_root", default='datasets/MSCOCO', type=str,help="Path to MSCOCO")
    parser.add_argument("--train_list", default='configs/coco/train_id.txt', type=str,
                        help="Path to train_id of MSCOCO")
    
    parser.add_argument("--cam_out_dir", default="cam_mask", type=str, help="CAM seeds path")
    parser.add_argument("--seg_out_dir", default="pseudo_mask", type=str, help="pesudo mask path")
    
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    
    # hyper parameters settings
    parser.add_argument("--radius", default=5, type=int)
    parser.add_argument("--beta", default=11, type=int)
    parser.add_argument("--logt", default=7, type=int)
    parser.add_argument("--threshold", default=0.41, type=float, help='the optimal one obtained for seeds')
    args = parser.parse_args()
    return args


def print_args(args):
    import pprint
    args_dict = vars(args)
    pprint.pprint(args_dict)
    
    
def put_palette(seg_label, out_name):
    out = seg_label.astype(np.uint8)
    out = Image.fromarray(out, mode='P')
    out.save(out_name)

    
def build_infer_dataset(args):
    infer_dataset = COCOImageDataset(
        img_name_list_path=args.train_list, 
        coco_root=args.coco_root,
        transform=torchvision.transforms.Compose(
            [np.asarray,
             Normalize(),
             imutils.HWC_to_CHW]))
    return infer_dataset # name, img-> C x H x W


def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(
        databin, 
        shuffle=False, 
        num_workers=args.num_workers // n_gpus, 
        pin_memory=False) 
    
    stride = 8
    
    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        for iter_, (name, img) in enumerate(
                tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            name = name[0]
            
            # if osp.exists(osp.join(args.seg_out_dir, name + '.png')):
            #     continue
            
            original_shape = img.shape
            min_size = 2 * args.radius * stride
            
            if original_shape[3] < min_size:
                pad_w = min_size - img.shape[3]
            else:
                pad_w = int(np.ceil(img.shape[3] / stride) * stride) - img.shape[3]
            
            if original_shape[2] < min_size:
                pad_h = min_size - img.shape[2]
            else:
                pad_h = int(np.ceil(img.shape[2] / stride) * stride) - img.shape[2]
                    
            p2d = (0, pad_w, 0, pad_h)
            
            img = F.pad(img, p2d)   
            dheight = int(np.ceil(img.shape[2] / stride))
            dwidth = int(np.ceil(img.shape[3] / stride))
            
            cam_dict = np.load(osp.join(args.cam_out_dir, name + '.npy'), allow_pickle=True).item()
            
            if len(tuple(cam_dict.keys())) == 0:
                cam = np.zeros((original_shape[2], original_shape[3]), np.uint8)
                put_palette(cam, osp.join(args.seg_out_dir, name + '.png'))
                continue
            
            cam_full_arr = np.zeros((args.num_classes, original_shape[2], original_shape[3]), np.float32)
            for k, v in cam_dict.items():
                cam_full_arr[k + 1] = v

            cam_full_arr[0] = args.threshold
            cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

            with torch.no_grad():
                aff_mat = torch.pow(model.forward(img.cuda(), to_dense=True), args.beta)
                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                for _ in range(args.logt):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                cam_full_arr = torch.from_numpy(cam_full_arr)
                cam_full_arr = F.avg_pool2d(cam_full_arr, stride, stride)
            
                cam_vec = cam_full_arr.view(args.num_classes, -1)
                cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
                cam_rw = cam_rw.view(1, args.num_classes, dheight, dwidth)
                cam_rw = F.interpolate(
                    input=cam_rw,
                    size=(img.shape[2], img.shape[3]), 
                    mode='bilinear', 
                    align_corners=False)
                _, cam_rw_pred = torch.max(cam_rw, 1)
                    
                res = np.uint8(cam_rw_pred.cpu().data[0])[:original_shape[2], :original_shape[3]]
                put_palette(res, os.path.join(args.seg_out_dir, name + '.png'))

            if process_id == n_gpus - 1 and iter_ % (len(databin) // 20) == 0:
                print("%d " % ((5*iter_+1)//(len(databin) // 20)), end='')


if __name__ == '__main__':
    args = get_args_parser()
    args.num_classes = 81
    args.cam_out_dir = osp.join(args.work_space, args.cam_out_dir)
    args.seg_out_dir = osp.join(args.work_space, args.seg_out_dir)
    print_args(args)
    Path(args.seg_out_dir).mkdir(parents=True, exist_ok=True)
    
    model = Net()
    model_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(model_dict)
    model.eval()
    model.cuda()

    n_gpus = torch.cuda.device_count()
    
    dataset = build_infer_dataset(args)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, 
                          args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()
