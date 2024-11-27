#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-30

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data


class _BaseDataset(data.Dataset):
    """
    Base dataset class
    """
    def __init__(
        self,
        root,
        split,
        ignore_label=255,
        is_train=True,
        base_size=None,
        crop_size=321,
        scales=(0.7, 1.3),
    ):
        self.root = root
        self.split = split
        self.ignore_label = ignore_label
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.is_train = is_train
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.files = []
        self._set_files()

        cv2.setNumThreads(0)

    def _set_files(self):
        """
        Create a file path/image id list.
        """
        raise NotImplementedError()

    def _load_data(self, image_id):
        """
        Load the image and label in numpy.ndarray
        """
        raise NotImplementedError()

    def _augmentation(self, image, label):
        # Scaling
        h, w = label.shape
        if self.base_size:
            if h > w:
                h, w = (self.base_size, int(self.base_size * w / h))
            else:
                h, w = (int(self.base_size * h / w), self.base_size)
        scale_factor = np.random.uniform(self.scales[0], self.scales[1])
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int32)

        # Random flipping
        if random.random() < 0.5:
            image = np.fliplr(image).copy() 
            label = np.fliplr(label).copy()
            
        # Normalize the image using broadcasting
        image = (image / 255.0 - self.mean) / self.std
        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        
        return image, label

    def __getitem__(self, index):
        image_id, image, label = self._load_data(index)
        if self.is_train:
            image, label = self._augmentation(image, label)
            image = image.transpose((2, 0, 1)) # HWC->CHW
            return image_id, image.astype(np.float32), label.astype(np.int64)
        
        else:
            image_tensor = (image / 255.0 - self.mean) / self.std  # Normalize for validation
            image_tensor = image_tensor.transpose((2, 0, 1)).astype(np.float32) # HWC->CHW
            original_image = image.astype(np.uint8)
            return image_id, original_image, image_tensor, label.astype(np.int64)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
