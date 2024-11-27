import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import os.path as osp
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from dataloaders.crf_utils import refine_crf_cam


TRAIN_FOLDER_NAME = "train2014"
VAL_FOLDER_NAME = "val2014"
ANNOT_FOLDER_NAME = "Annotations"
LABEL_FOLDER_NAME = 'ImageLabel'


def load_image_label_list_from_npy_coco(coco_root, img_name_list):
    label_file_path = osp.join(coco_root, LABEL_FOLDER_NAME, 'COCO_cls_labels.npy')
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    label_list = [cls_labels_dict[img_name + '.jpg'] for img_name in img_name_list]   
    return label_list


def load_image_label_list_from_npy(coco_root, img_name_list):
    label_file_path = osp.join(coco_root, LABEL_FOLDER_NAME, 'COCO_cls_labels.npy')
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_path(img_name, coco_root, is_train=True):
    if is_train:
        return os.path.join(coco_root, TRAIN_FOLDER_NAME, img_name + '.jpg')
    else:
        return os.path.join(coco_root, VAL_FOLDER_NAME, img_name + '.jpg')


def load_img_name_list(dataset_path):
    img_name_list = open(dataset_path).read().splitlines()
    return img_name_list


class COCOImageDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, transform=None):
        if 'train' in img_name_list_path:
            self.train = True
        elif 'val' in img_name_list_path:
            self.train = False
        else:
            raise NotImplementedError
        
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.coco_root = coco_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = Image.open(get_img_path(name, self.coco_root, is_train=self.train)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return name, img


class COCOClsDataset(COCOImageDataset):
    def __init__(self, img_name_list_path, coco_root, transform=None):
        super().__init__(img_name_list_path, coco_root, transform)
        self.label_list = load_image_label_list_from_npy(coco_root, self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)
        label = torch.from_numpy(self.label_list[idx])
        return name, img, label


class COCOClsDatasetMSF(COCOClsDataset):
    def __init__(self, img_name_list_path, coco_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, coco_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label


class ExtractAffinityLabelInRadius():
    def __init__(self, cropsize, radius=5):
        self.radius = radius
        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius - 1
        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):
        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])
        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 81), np.less(labels_from, 81))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)
        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)
        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)
        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)


class COCOAffDatasetCRF(COCOImageDataset):
    def __init__(self, img_name_list_path, cam_npy_dir, cropsize, coco_root, 
                 low_alpha=1, high_alpha=3, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, coco_root, transform=None)
        self.cam_dir = cam_npy_dir
        self.low_alpha = low_alpha
        self.high_alpha = high_alpha
        self.coco_root = coco_root
        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list
        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(
            cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)
    
    def transform_img_label(self, img, label):
        for joint_transform, img_transform, label_transform \
            in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):
            if joint_transform: # RandomCrop, RandomHorizontalFlip
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]
            if img_transform: # ColorJitter, np.asarray, RandomCrop, normalize, HWC_to_CHW
                img = img_transform(img)
            if label_transform: # AvgPool2d
                label = label_transform(label)
            
        return img, label
                
    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)
        label_cam_path = os.path.join(self.cam_dir, f"{name}.npy")
        cam_dict = np.load(label_cam_path, allow_pickle=True).item()
        original_image = np.array(img)
        
        if len(tuple(cam_dict.keys())) == 0:
            label_la = {0: np.zeros((img.height, img.width))}
            label_ha = label_la.copy()
            label = np.array(list(label_la.values()) + list(label_ha.values()))
            label = np.transpose(label, (1, 2, 0)) 
            img, label = self.transform_img_label(img, label)
            label_la, label_ha = np.array_split(label, 2, axis=-1)
            label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
            label = label_la.copy()
            
        else:
            label_la, label_ha = refine_crf_cam(
                cam_dict=cam_dict, 
                original_image=original_image, 
                low_alpha=self.low_alpha, 
                high_alpha=self.high_alpha)
            
            label = np.array(list(label_la.values()) + list(label_ha.values()))
            label = np.transpose(label, (1, 2, 0)) # H, W, Cls
            
            img, label = self.transform_img_label(img, label)
            
            no_score_region = np.max(label, -1) < 1e-5
            label_la, label_ha = np.array_split(label, 2, axis=-1)
            label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
            label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
            label = label_la.copy()
            label[label_la == 0] = 255
            label[label_ha == 0] = 0
            label[no_score_region] = 255 # mostly outer of cropped region
            
        label = self.extract_aff_lab_func(label)
        
        return img, label
    

class COCOSegmentationLabelDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 id_list_file="configs/coco/train_id.txt",
                 annotation_dir='MaskSets'):
        super(COCOSegmentationLabelDataset, self).__init__()

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        if "train" in id_list_file:
            self.image_dir = osp.join(data_dir, TRAIN_FOLDER_NAME)
            self.mask_dir = osp.join(data_dir, annotation_dir, TRAIN_FOLDER_NAME)
        else:
            self.image_dir = osp.join(data_dir, VAL_FOLDER_NAME)
            self.mask_dir = osp.join(data_dir, annotation_dir, VAL_FOLDER_NAME)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name_id = self.ids[idx]
        image_path = osp.join(self.image_dir, name_id + '.jpg')
        label_path = osp.join(self.mask_dir, name_id + '.png')
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        
        return {"name_id": name_id, "image": image, "label": label}