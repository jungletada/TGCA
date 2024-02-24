import os
import torch
import argparse
import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

    
def run(args):
    dataset = VOCSemanticSegmentationDataset(
        split=args.chainer_eval_set, 
        data_dir=args.voc12_root)
    
    def eval_curve(threshold):
        preds = []
        labels = []
        miou = 0.
        for i, img_id in enumerate(dataset.ids):
            cam_dict = np.load(os.path.join(args.eval_cam_dir, img_id + '.npy'), allow_pickle=True).item()
            
            cams = np.stack(list(cam_dict.values()), axis=0) # (#val_cls, H, W)
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            
            keys = (torch.stack(tuple(cam_dict.keys())) + 1).numpy()
            keys = np.pad(keys, (1, 0), mode='constant')
            
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            
            preds.append(cls_labels.copy())
            labels.append(dataset.get_example_by_keys(i, (1,))[0])

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator
        miou = np.nanmean(iou)
        print("threshold: {:.2f}, miou: {:.4f}".format(threshold, miou))
        # print('among_pred_fg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
        return miou
    
    best_res = 0.
    best_threshold = 0
    for t in range(30, 60):
        miou = eval_curve(t / 100.)
        if miou > best_res: 
            best_res = miou
            best_threshold = t / 100.
        else:
            break
            
    print("-"*30)
    print("Best threshold: {}, best miou: {:.4f}, num_imgs: {}".format(
        best_threshold, best_res, len(dataset.ids)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate CAMs', add_help=False)
    parser.add_argument("--chainer_eval_set", default="train", type=str)
    parser.add_argument('--voc12_root', default='datasets/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument('--eval_cam_dir', default='results_voc/MCTG/cam_mask', help='cam_mask directory')
    args = parser.parse_args()
    
    run(args=args)
    
    