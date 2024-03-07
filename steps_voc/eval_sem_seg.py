import os, sys
import argparse
import numpy as np
import os.path as osp
import imageio.v2 as imageio

sys.path.append(osp.dirname(__file__) + os.sep + '../')
from data.voc12.dataloader import VOCSegmentationLabelDataset
from engine import calc_semantic_segmentation_confusion


def run(args):
    dataset = VOCSegmentationLabelDataset(
        split=args.chainer_eval_set, 
        data_dir=args.voc12_root)
    
    preds = []
    labels = []
    
    for i, id_ in enumerate(dataset.ids):
        cls_labels = imageio.imread(os.path.join(args.seg_out_dir, id_ + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        preds.append(cls_labels.copy())
        labels.append(dataset[i]["label"])

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print("False Positive: {:.4f}, False Negative: {:.4f}".format(fp[0], fn[0]))
    # print("IoU(%) for each class:")
    # for res in iou:
    #     print("{:.2f}".format(res * 100.))
    print("mIoU (%): {:.2f}".format(np.nanmean(iou) * 100.))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate Pseudo Masks', add_help=False)
    parser.add_argument("--work_space", default="results/MCTG", type=str)
    parser.add_argument("--chainer_eval_set", default="train", type=str)
    parser.add_argument('--voc12_root', default='datasets/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument('--seg_out_dir', default='pseudo_mask', help='pseudo_mask directory')
    args = parser.parse_args()
    
    args.seg_out_dir = osp.join(args.work_space, args.seg_out_dir)
    run(args=args)
