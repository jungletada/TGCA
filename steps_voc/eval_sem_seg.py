import os
import torch
import argparse
import imageio
import numpy as np

from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion


def run(args):
    dataset = VOCSemanticSegmentationDataset(
        split=args.chainer_eval_set, 
        data_dir=args.voc12_root)

    preds = []
    labels = []
    n_img = 0
    
    for i, id_ in enumerate(dataset.ids):
        cls_labels = imageio.imread(os.path.join(args.sem_seg_out_dir, id_ + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())
        labels.append(dataset.get_example_by_keys(i, (1,))[0])
        n_img += 1

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator
    print("Total images=", n_img)
    print("False Positive: {:.4f}, False Negative: {:.4f}".format(fp[0], fn[0]))
    # print("IoU(%) for each class:")
    # for res in iou:
    #     print("{:.2f}".format(res * 100.))
    print("mIoU (%): {:.2f}".format(np.nanmean(iou) * 100.))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate Pseudo Masks', add_help=False)
    parser.add_argument("--chainer_eval_set", default="train", type=str)
    parser.add_argument('--voc12_root', default='datasets/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument('--sem_seg_out_dir', default='results/MCTG/pseudo_mask', help='pseudo_mask directory')
    args = parser.parse_args()
    
    run(args=args)
