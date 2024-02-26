import sys
import torch
import torchvision
import os
import os.path as osp
import argparse
import importlib
from tqdm import tqdm
import numpy as np
from torch.backends import cudnn
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
import PIL.Image as Image
from pathlib import Path

sys.path.append(osp.dirname(__file__) + os.sep + '../')
from psa_network.tool import pyutils, imutils
from misc import torchutils
from psa_network.network.transform import Normalize
from psa_network.network.resnet38_aff import ResNet38d_Aff
from data.voc12.dataloader_psa import VOC12ImageDataset
cudnn.enabled = True


def get_args_parser():
    parser = argparse.ArgumentParser()
    # checkpoint and data settings
    parser.add_argument("--work_space", default="results/MCTG", type=str)
    parser.add_argument("--voc12_root", default='datasets/VOCdevkit/VOC2012/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_aff", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument('--infer_list', default='configs/voc12/train_aug.txt', type=str, help='image list path')
    parser.add_argument("--cam_out_dir", required=True, type=str, help="CAM seeds path")
    parser.add_argument("--seg_out_dir", required=True, type=str, help="pesudo mask path")
    # hyper parameters settings
    parser.add_argument("--beta", default=11, type=int)
    parser.add_argument("--logt", default=7, type=int)
    parser.add_argument("--threshold", default=0.41, type=float, help='the optimal one obtained for seeds')
    args = parser.parse_args()
    return args


def put_palette(seg_label, out_name):
    out = seg_label.astype(np.uint8)
    out = Image.fromarray(out, mode='P')
    # out.putpalette(data_voc.palette)
    out.save(out_name)


def print_args(args):
    import pprint
    args_dict = vars(args)
    pprint.pprint(args_dict)
    
    
def build_infer_dataset(args):
    infer_dataset = VOC12ImageDataset(
        args.infer_list, 
        voc12_root=args.voc12_root,
        transform=torchvision.transforms.Compose(
            [np.asarray,
             Normalize(),
             imutils.HWC_to_CHW]))
    return infer_dataset # name, img-> C x H x W


def build_infer_dataloader(args, infer_dataset):
    infer_data_loader = DataLoader(
        infer_dataset, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=False) 
    
    return infer_data_loader
    

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
            original_shape = img.shape
            padded_size = (int(np.ceil(img.shape[2] / stride) * stride), int(np.ceil(img.shape[3] / stride) * stride))
            p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
            img = F.pad(img, p2d)

            dheight = int(np.ceil(img.shape[2] / stride))
            dwidth = int(np.ceil(img.shape[3] / stride))

            cam = np.load(osp.join(args.cam_out_dir, name + '.npy'), allow_pickle=True).item()

            cam_full_arr = np.zeros((args.num_classes, original_shape[2], original_shape[3]), np.float32)
            for k, v in cam.items():
                cam_full_arr[k + 1] = v

            cam_full_arr[0] = args.threshold
            cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

            aff_mat = torch.pow(model.forward(img.cuda(), True), args.beta)
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
            
            put_palette(res, osp.join(args.seg_out_dir, name + '.png'))
    
    
if __name__ == '__main__':
    args = get_args_parser()
    args.num_classes = 21
    args.cam_out_dir = osp.join(args.work_space, args.cam_out_dir)
    args.seg_out_dir = osp.join(args.work_space, args.seg_out_dir)
    print_args(args)
    Path(args.seg_out_dir).mkdir(parents=True, exist_ok=True)
    
    model = ResNet38d_Aff()
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
