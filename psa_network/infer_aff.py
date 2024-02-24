import sys
import torch
import torchvision

import os.path
import argparse
import importlib
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import PIL.Image as Image
from pathlib import Path

from network.transform import Normalize
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from data.voc12.dataloader_psa import VOC12ImageDataset
from tool import imutils


def get_args_parser():
    parser = argparse.ArgumentParser()
    # checkpoint and data settings
    parser.add_argument("--work_space", default="results/MCTG", type=str)
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


def get_indices_in_radius(height, width, radius):
    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius+1, radius):
            if x*x + y*y < radius*radius:
                search_dist.append((y, x))

    full_indices = np.reshape(np.arange(0, height * width, dtype=np.int64),
                              (height, width))
    radius_floor = radius-1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])

    indices_from_to_list = []

    for dy, dx in search_dist:

        indices_to = full_indices[dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_from_to = np.stack((indices_from, indices_to), axis=1)

        indices_from_to_list.append(indices_from_to)

    concat_indices_from_to = np.concatenate(indices_from_to_list, axis=0)

    return concat_indices_from_to


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
        pin_memory=True) 
    
    return infer_data_loader
    

def infer_psa(args):
    args.voc12_root = "datasets/VOCdevkit/VOC2012"
    args.num_classes = 21
    
    print_args(args)
    args.cam_out_dir = os.path.join(args.work_space, args.cam_out_dir)
    args.seg_out_dir = os.path.join(args.work_space, args.seg_out_dir)
    Path(args.seg_out_dir).mkdir(parents=True, exist_ok=True)
    
    model = getattr(importlib.import_module(args.network), 'Net')()
    model_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(model_dict)

    model.eval()
    model.cuda()

    infer_dataset = build_infer_dataset(args)
    infer_data_loader = build_infer_dataloader(args, infer_dataset)
    infer_bar = tqdm(infer_data_loader)

    stride = 8
    for iter, (name, img) in enumerate(infer_bar):
        name = name[0]
        original_shape = img.shape
        padded_size = (int(np.ceil(img.shape[2] / stride) * stride), int(np.ceil(img.shape[3] / stride) * stride))
        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2] / stride))
        dwidth = int(np.ceil(img.shape[3] / stride))

        cam = np.load(os.path.join(args.cam_out_dir, name + '.npy'), allow_pickle=True).item()

        cam_full_arr = np.zeros((args.num_classes, original_shape[2], original_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k + 1] = v

        cam_full_arr[0] = args.threshold
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        with torch.no_grad():
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
            
            put_palette(res, os.path.join(args.seg_out_dir, name + '.png'))
    
    
if __name__ == '__main__':
    args = get_args_parser()
    print_args(args)
    infer_psa(args)
