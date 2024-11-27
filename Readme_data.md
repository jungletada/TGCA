# VOC2012 and MSCOCO Dataset
This is an instruction for setting up `PASCAL VOC` and `MSCOCO` dataset. This "Readme" is refered to [kazuto1011
/
deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch/tree/master/data/datasets).
## PASCAL VOC 2012
1. Download PASCAL VOC 2012. Put `VOC2012` into the `data` folder.

```sh
$ bash scripts/setup_voc12.sh [PATH TO DOWNLOAD]
```

```
data
└── VOCdevkit
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        │   └── Segmentation
        ├── JPEGImages
        ├── SegmentationObject
        └── SegmentationClass
```
2. Add SBD augmentated training data as `SegmentationClassAug`.


* Convert by yourself ([here](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal)).
* Or download pre-converted files ([here](https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation)).

3. Download official image sets as `ImageSets/SegmentationAug`.

* From https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
* Or https://github.com/kazuto1011/deeplab-pytorch/files/2945588/list.zip

```sh
data
└── VOCdevkit
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        │   └── Segmentation 
        ├── JPEGImages
        ├── SegmentationObject
        ├── SegmentationClass
        └── SegmentationClassAug # ADDED!!
            └── 2007_000032.png
```

## COCO-Stuff 10k

### Setup

1. Run the script below to download the dataset (2GB).

```sh
$ bash ./scripts/setup_cocostuff10k.sh [PATH TO DOWNLOAD]
```

2. Put `MSCOCO` into the `datasets` folder.

### Dataset structure

```sh
data
└──MSCOCO
    ├── images
    │   ├── COCO_train2014_000000000077.jpg
    │   └── ...
    ├── annotations
    │   ├── instances_train2014.json
    │   └── ...
```