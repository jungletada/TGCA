import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from tool import flip, scale_img, scale_gt, RandomCrop, crop
import PIL.Image as Image

mean = np.expand_dims(np.array([0.485, 0.456, 0.406]), axis=(0, 1))
std = np.expand_dims(np.array([0.229, 0.224, 0.225]), axis=(0, 1))

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]


classes = np.array((
    'background', 
    'aeroplane', 
    'bicycle', 
    'bird', 
    'boat',
    'bottle', 
    'bus', 
    'car', 
    'cat', 
    'chair',
    'cow', 
    'diningtable', 
    'dog', 
    'horse',
    'motorbike', 
    'person', 
    'pottedplant',
    'sheep', 
    'sofa', 
    'train', 
    'tvmonitor'
))


class VOCSegmentationDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.img_path = args.img_path
        self.crop_size = args.crop_size
        self.scale = np.random.uniform(0.7, 1.3)
    
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        piece = img_name.replace('.jpg', '').strip()
        flip_p = np.random.uniform(0, 1)
        
        input_img = cv2.imread(os.path.join(self.img_path, piece + '.jpg'))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float)
        input_img = scale_img(input_img, self.scale)
        input_img = flip(input_img, flip_p)
        input_img = (input_img / 255. - mean) / std
        input_img, img_hp, img_wp = crop(input_img, self.crop_size)
        
        pseudo_label = np.asarray(Image.open(
            os.path.join(self.args.seg_pgt_path, piece + '.png')))
        pseudo_label = scale_gt(pseudo_label, self.scale)
        pseudo_label = flip(pseudo_label, flip_p)
        pseudo_label = crop(pseudo_label, self.crop_size, False, img_hp, img_wp)[0]
        
        original_img = np.zeros_like(input_img)
        original_img = (input_img * std + mean) * 255.
        original_img = original_img.astype(np.uint8)
        
        input_img = input_img.transpose((2, 0, 1))
        input_img = torch.from_numpy(input_img).float()
        pseudo_label = torch.from_numpy(pseudo_label).float()
        original_img = original_img.transpose((2, 0, 1))
        return input_img, original_img, pseudo_label, piece
        
    def __len__(self):
        return len(self.img_path)