import os
import sys
import cv2
import torch
import imageio.v2 as imageio
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__) + os.sep + '../..')
from misc import imutils
from data.base_seg_dataset import _BaseDataset

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
LABEL_FOLDER_NAME = "ImageLabel"
IGNORE = 255


CAT_LIST = ['background','aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

N_CAT = len(CAT_LIST)

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,
        64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,
        0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,
        64,64,0,  192,64,0,  64,192,0, 192,192,0]

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))


def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    elem_list = minidom.parse(os.path.join(
        voc12_root, 
        ANNOT_FOLDER_NAME, 
        decode_int_filename(img_name) + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):
    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list, voc12_root):
    cls_labels_path = os.path.join(voc12_root, LABEL_FOLDER_NAME, 'cls_labels.npy')
    cls_labels_dict = np.load(cls_labels_path, allow_pickle=True).item()
    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])


def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def convert_to_int(s):
    return int(s.decode('utf-8').replace('_', ''))


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.int32, converters={0: convert_to_int})
    return img_name_list


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class VOC12ImageDataset(Dataset):
    def __init__(self, 
                 img_name_list_path, 
                 voc12_root,
                 resize_long=None, 
                 rescale=None, 
                 img_normal=TorchvisionNormalize(), 
                 hor_flip=False,
                 crop_size=None, 
                 crop_method=None, 
                 to_torch=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return {'name': name_str, 'img': img}


class VOC12ClassificationDataset(VOC12ImageDataset):
    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), 
                 hor_flip=False, crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(voc12_root, self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out['label'] = torch.from_numpy(self.label_list[idx])

        return out


class VOC12ClassificationDataset_Single(VOC12ImageDataset):
    def __init__(self, img_name_list_path, voc12_root, 
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        
        self.label_list = load_image_label_list_from_npy(voc12_root, self.img_name_list)
        self.len = np.sum(self.label_list).astype(np.int)
        self.idx_map = np.zeros(self.len,dtype=np.int)
        self.bias = np.zeros(self.len,dtype=np.int)
        print('single_obj_data_num:',self.len)
        idx = 0
        for i in range(len(self.label_list)):
            x = np.sum(self.label_list[i])
            while x > 0:
                x = x-1
                self.idx_map[idx] = i
                self.bias[idx] = x
                idx = idx + 1
        print(idx)
        # print(self.bias[:30])
    def __getitem__(self, idx):
        if idx < len(self.img_name_list):
            out = super().__getitem__(idx)
            out['label'] = torch.from_numpy(self.label_list[idx])
        else:
            idx = idx%len(self.label_list)
            bias = self.bias[idx]
            idx = self.idx_map[idx]
            label = torch.from_numpy(self.label_list[idx])
            label = torch.nonzero(label)[:,0][bias]


            name = self.img_name_list[idx]
            name_str = decode_int_filename(name)

            mask = imageio.imread(os.path.join(self.voc12_root, 'SegmentationClassAug', name_str + '.png'))
            img0 = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))
            # print(img0.dtype)
            # print(img)
            mask = np.stack([mask,mask,mask],axis=2)
            mask = (mask==0)*1 + (mask==(label+1).item())*1
            img_rand = np.random.randint(255, size=img0.shape)
            # wh = img0.shape[:2]
            # img_rand = np.stack([torch.ones(wh)*124,torch.ones(wh)*116,torch.ones(wh)*104],axis=2)
            img = (mask*img0+(1-mask)*img_rand).astype(np.uint8)

            if self.resize_long:
                img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

            if self.rescale:
                img = imutils.random_scale(img, scale_range=self.rescale, order=3)

            if self.img_normal:
                img = self.img_normal(img)

            if self.hor_flip:
                img = imutils.random_lr_flip(img)

            if self.crop_size:
                if self.crop_method == "random":
                    img = imutils.random_crop(img, self.crop_size, 0)
                else:
                    img = imutils.top_left_crop(img, self.crop_size, 0)

            if self.to_torch:
                img = imutils.HWC_to_CHW(img)
            out = {'name': name_str, 'img': img, 'label':F.one_hot(label, num_classes=20).type(torch.float32)}
        return out

    def __len__(self):
        print('len:',self.len + len(self.img_name_list))
        return self.len + len(self.img_name_list)


class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):
    def __init__(self, img_name_list_path, voc12_root, 
                 img_normal=TorchvisionNormalize(), scales=(1.0,), make_seg=False):
        super().__init__(img_name_list_path, voc12_root, img_normal=img_normal)
        if isinstance(scales, int) or isinstance(scales, float):
            scales = tuple([scales])
        self.scales = scales # (1.0, 0.5, 1.5, 2.0)
        self.make_seg = make_seg
        
    def __getitem__(self, idx):
        """
        Return: pack: dict
        name: image name: str, 
        img: multi-scale image list->[(2 x 3 x W' x H')], length=#scales
        size: original image shape: Tuple (W x H)
        label: one-hot label: Tensor->[num_cls]
        """
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        img = imageio.imread(get_img_path(name_str, self.voc12_root))

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
            
        if len(self.scales) == 1 and self.make_seg:
            ms_img_list = ms_img_list[0]

        out = {"name": name_str, 
               "img": ms_img_list, 
               "size": (img.shape[0], img.shape[1]),
               "label": torch.from_numpy(self.label_list[idx])}
        return out
        
        
class VOCAugSegmentationDataset(_BaseDataset):
    """
    PASCAL VOC Segmentation dataset with extra annotations
    """
    def __init__(self,
                 voc12_root="datasets/VOCdevkit/VOC2012",
                 pseudo_dir=None,  
                 **kwargs):
        self.root = voc12_root
        self.img_dir = "JPEGImages"
        self.label_dir = "SegmentationClassAug"
        self.pseudo_dir = self.label_dir if pseudo_dir is None else pseudo_dir
        super(VOCAugSegmentationDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ["train", "train_aug"]:
            id_list = osp.join(
                self.root, "ImageSets/SegmentationAug", self.split + "_id.txt"
            )
            id_list = tuple(open(id_list, "r"))
            id_list = [id_.rstrip() for id_ in id_list]
            self.files = [f"/{self.img_dir}/{id_}.jpg" for id_ in id_list]
            self.labels = [f"/{self.pseudo_dir}/{id_}.png" for id_ in id_list]
            
        elif self.split in ["val"]:
            id_list = osp.join(
                self.root, "ImageSets/SegmentationAug", self.split + "_id.txt"
            )
            id_list = tuple(open(id_list, "r"))
            id_list = [id_.rstrip() for id_ in id_list]
            self.files = [f"/{self.img_dir}/{id_}.jpg" for id_ in id_list]
            self.labels = [f"/{self.label_dir}/{id_}.png" for id_ in id_list]
            
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index].split("/")[-1].split(".")[0]
        image_path = osp.join(self.root, self.files[index][1:])
        label_path = osp.join(self.root, self.labels[index][1:])
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image_id, image, label


class VOCSegmentationLabelDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        super(VOCSegmentationLabelDataset, self).__init__()

        if split not in ['train', 'trainval', 'val']:
            raise ValueError(
                'please pick split from \'train\', \'trainval\', \'val\'')
            
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Segmentation/{0}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name_id = self.ids[idx]
        image_path = osp.join(self.data_dir, "JPEGImages", name_id + '.jpg')
        label_path = osp.join(self.data_dir, "SegmentationClass", name_id + '.png')
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        label[label==255] = -1
        return {"image": image, "label": label}
    
    
if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torchvision
    import yaml
    from torchvision.utils import make_grid
    from tqdm import tqdm

    kwargs = {"nrow": 5, "padding": 50}
    batch_size = 25

    # dataset = VOCAugSegmentationDataset(
    #     root="datasets/VOCdevkit/VOC2012",
    #     split="train_aug",
    #     pseudo_dir=None,
    #     ignore_label=255,
    #     is_train=True,
    #     base_size=None,
    #     crop_size=321,
    #     scales=(0.7, 1.3))
    # print(dataset)
    # from torch.utils import data
    # loader = data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=True)
    
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    # for i, (image_ids, images, labels) in tqdm(
    #     enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False):
        
    #     if i == 0:
    #         print(f"Input size:{images.shape}; Label size: {labels.shape}")
           
    #         images = (images * std + mean) * 255.0
    #         images = images.int()
        
    #         image = make_grid(images, pad_value=-1, **kwargs).numpy()
    #         image = np.transpose(image, (1, 2, 0))
    #         mask = np.zeros(image.shape[:2])
    #         mask[(image != -1)[..., 0]] = 255
    #         image = np.dstack((image, mask)).astype(np.uint8)
            
    #         labels = labels[:, np.newaxis, ...]
    #         label = make_grid(labels, pad_value=255, **kwargs).numpy()
    #         label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
    #         label = cm.jet_r(label_ / 21.0) * 255
    #         mask = np.zeros(label.shape[:2])
    #         label[..., 3][(label_ == 255)] = 0
    #         label = label.astype(np.uint8)

    #         tiled_images = np.hstack((image, label))
    #         tiled_images = cv2.cvtColor(tiled_images, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite("voc12.png", tiled_images)
    #         # plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
    #         # plt.show()
    #         break
    
    # dataset = VOCAugSegmentationDataset(
    #     root="datasets/VOCdevkit/VOC2012",
    #     split="val",
    #     pseudo_dir=None,
    #     ignore_label=255,
    #     is_train=False)
    
    # name, original_image, image_tensor, label = dataset[60]
    # print(f"image_id: {name}")
    # print(f"original: {original_image.shape}")
    # print(f"tensor: {image_tensor.shape}")
    # print(f"label: {label.shape}")
    
    dataset = VOCSegmentationLabelDataset(
        data_dir="datasets/VOCdevkit/VOC2012",
        split='train'
    )
    print(dataset[10]["image"].shape)
    
    print(dataset[10]["label"].shape)
    print(np.max(dataset[10]["label"]))