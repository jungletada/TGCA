import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np
import PIL.Image


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]
    return img_name_list


def load_image_label_list_from_npy_voc(img_name_list):
    label_file_path = 'configs/voc12/cls_labels.npy'
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = [cls_labels_dict[img_name] for img_name in img_name_list]
    return label_list


def load_image_label_list_from_npy_coco(img_name_list):
    label_file_path = 'configs/coco/COCO_cls_labels.npy'
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = [cls_labels_dict[img_name + '.jpg'] for img_name in img_name_list]   
    return label_list


class COCOClsDataset(Dataset):
    def __init__(self, coco_root, train=True, transform=None):
        img_name_list_path = os.path.join('configs/coco', f'{"train" if train else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy_coco(self.img_name_list)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.train:
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_name_list)


class COCOClsDatasetMS(Dataset):
    def __init__(self, coco_root, scales, train=True, transform=None, unit=1):
        img_name_list_path = os.path.join(
            'configs/coco', f'{"train" if train else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy_coco(self.img_name_list)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.unit = unit
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.train:
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
        
        label = torch.from_numpy(self.label_list[idx])
        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))
        
        msf_img_list = []
        for s in self.scales: 
            if s == 1:
                s_img = img
            else:
                target_size = (round(rounded_size[0] * s), round(rounded_size[1] * s))
                s_img = img.resize(target_size, resample=PIL.Image.BICUBIC)
            
            s_img = self.transform(s_img)
            flip_img_pair = torch.stack([s_img, torch.flip(s_img, [-1])], dim=0)
            msf_img_list.append(flip_img_pair)
        
        out = {"name": name, 
               "img": msf_img_list, 
               "size": (img.height, img.width),
               "label": label}
        return out
    
    def __len__(self):
        return len(self.img_name_list)


class VOC12Dataset(Dataset):
    def __init__(self, voc12_root, infer_list, transform=None):
        self.img_name_list = load_img_name_list(infer_list)
        self.label_list = load_image_label_list_from_npy_voc(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_name_list)


class VOC12DatasetMS(Dataset):
    def __init__(self, voc12_root, infer_list, scales, transform=None, unit=1):
        self.img_name_list = load_img_name_list(infer_list)
        self.label_list = load_image_label_list_from_npy_voc(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform
        self.unit = unit
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        
        label = torch.from_numpy(self.label_list[idx])
        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), 
                        int(round(img.size[1] / self.unit) * self.unit))
        
        msf_img_list = []
        for s in self.scales: 
            if s == 1:
                s_img = img
            else:
                target_size = (round(rounded_size[0] * s), round(rounded_size[1] * s))
                s_img = img.resize(target_size, resample=PIL.Image.BICUBIC)
            
            s_img = self.transform(s_img)
            flip_img_pair = torch.stack([s_img, torch.flip(s_img, [-1])], dim=0)
            msf_img_list.append(flip_img_pair)
        
        out = {"name": name, 
               "img": msf_img_list, 
               "size": (img.height, img.width),
               "label": label}
        return out

    def __len__(self):
        return len(self.img_name_list)


def build_transform(is_train, make_cam, args):
    resize_img = args.input_size > 32
    # if args.train_interpolation == 'bicubic':
    #     interpolation=transforms.InterpolationMode.BICUBIC
    # elif args.train_interpolation == 'bilinear':
    #     interpolation=transforms.InterpolationMode.BILINEAR
    # else:
    #     interpolation=transforms.InterpolationMode.NEAREST
    
    interpolation=transforms.InterpolationMode.BICUBIC
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_img:
            # replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    pipeline = []
    if resize_img and not make_cam:
        size = int((256 / 224) * args.input_size)
        # to maintain same ratio w.r.t. 224 images
        pipeline.append(transforms.Resize(
            size=size, 
            interpolation=interpolation))
        pipeline.append(transforms.CenterCrop(args.input_size))

    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(pipeline)


def build_dataset(is_train, make_cam, args):
    transform = build_transform(is_train, make_cam, args)
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12':
        dataset = VOC12Dataset(
            voc12_root=args.voc12_root,
            infer_list='configs/voc12/train_aug_id.txt',
            transform=transform)
        nb_classes = 20

    elif args.data_set == 'VOC12MS':
        dataset = VOC12DatasetMS(
            voc12_root=args.voc12_root,
            infer_list=args.infer_list,
            scales=tuple(args.scales),
            transform=transform)
        nb_classes = 20

    elif args.data_set == 'COCO':
        dataset = COCOClsDataset(
            coco_root='datasets/MSCOCO', 
            train=is_train, 
            transform=transform)
        nb_classes = 80

    elif args.data_set == 'COCOMS':
        dataset = COCOClsDatasetMS(
            coco_root='datasets/MSCOCO', 
            scales=tuple(args.scales), 
            train=is_train, 
            transform=transform)
        nb_classes = 80

    return dataset, nb_classes


